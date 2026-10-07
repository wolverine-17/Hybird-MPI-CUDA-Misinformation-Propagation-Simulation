#include <mpi.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "sbfc_core.hpp"
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

typedef struct { uint32_t n,degree,ticks,sample,source,preview,graphOnly; uint64_t seed; double alpha,beta,verify,forget; } Config;
typedef struct { uint64_t edges,*offset; uint32_t *adj; } Graph;

#define CUDA_OK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,2); } } while(0)
static void *xmalloc(size_t n){void*p=malloc(n);if(!p){fprintf(stderr,"Out of memory (%zu bytes)\n",n);MPI_Abort(MPI_COMM_WORLD,3);}return p;}
static Config args(int argc,char**argv){Config c={10000,6,300,1,0,10000,0,42,.3,.5,.05,.1};for(int i=1;i+1<argc;i+=2){char*k=argv[i],*v=argv[i+1];if(!strcmp(k,"--graph-only"))c.graphOnly=strtoul(v,0,10);else if(!strcmp(k,"--nodes"))c.n=strtoul(v,0,10);else if(!strcmp(k,"--mean-degree"))c.degree=strtoul(v,0,10);else if(!strcmp(k,"--ticks"))c.ticks=strtoul(v,0,10);else if(!strcmp(k,"--sample-every"))c.sample=strtoul(v,0,10);else if(!strcmp(k,"--source-node"))c.source=strtoul(v,0,10);else if(!strcmp(k,"--preview-nodes"))c.preview=strtoul(v,0,10);else if(!strcmp(k,"--alpha"))c.alpha=strtod(v,0);else if(!strcmp(k,"--beta"))c.beta=strtod(v,0);else if(!strcmp(k,"--verify"))c.verify=strtod(v,0);else if(!strcmp(k,"--forget"))c.forget=strtod(v,0);else if(!strcmp(k,"--seed"))c.seed=strtoull(v,0,10);}if(c.n<4||c.source>=c.n||c.sample<1){fprintf(stderr,"Invalid configuration\n");MPI_Abort(MPI_COMM_WORLD,4);}if(c.preview>c.n)c.preview=c.n;return c;}

static Graph make_graph(uint32_t n,uint32_t degree,uint64_t seed){uint32_t m=degree/2;if(m<1)m=1;if(m>=n)m=n-1;uint32_t first=m+1;uint64_t cap=(uint64_t)first*(first-1)/2+(uint64_t)(n-first)*m,e=0,pooln=0;uint32_t*u=(uint32_t*)xmalloc(cap*4),*v=(uint32_t*)xmalloc(cap*4),*pool=(uint32_t*)xmalloc(cap*8),*pick=(uint32_t*)xmalloc(m*4);for(uint32_t a=0;a<first;a++)for(uint32_t b=a+1;b<first;b++){u[e]=a;v[e++]=b;pool[pooln++]=a;pool[pooln++]=b;}uint64_t r=seed;for(uint32_t node=first;node<n;node++){for(uint32_t j=0;j<m;j++){uint32_t q;int dup;do{r=mix64(r);q=pool[r%pooln];dup=(q==node);for(uint32_t k=0;k<j;k++)dup|=pick[k]==q;}while(dup);pick[j]=q;u[e]=node;v[e++]=q;pool[pooln++]=node;pool[pooln++]=q;}}free(pool);free(pick);Graph g={e,(uint64_t*)calloc((size_t)n+1,8),(uint32_t*)xmalloc(e*8)};if(!g.offset)MPI_Abort(MPI_COMM_WORLD,3);for(uint64_t i=0;i<e;i++){g.offset[u[i]+1]++;g.offset[v[i]+1]++;}for(uint32_t i=1;i<=n;i++)g.offset[i]+=g.offset[i-1];uint64_t*cur=(uint64_t*)xmalloc(n*8);memcpy(cur,g.offset,n*8);for(uint64_t i=0;i<e;i++){g.adj[cur[u[i]]++]=v[i];g.adj[cur[v[i]]++]=u[i];}free(cur);free(u);free(v);return g;}

__global__ static void step_kernel(const uint64_t*off,const uint32_t*adj,const uint8_t*cur,uint8_t*next,uint32_t begin,uint32_t count,uint32_t tick,uint64_t seed,double alpha,double beta,double verify,double forget) {
    uint32_t local=blockIdx.x*blockDim.x+threadIdx.x;
    if(local<count)next[local]=update_state(off,adj,cur,local,begin+local,tick,seed,alpha,beta,verify,forget);
}

static void emit(uint32_t tick,uint64_t s,uint64_t b,uint64_t f,uint64_t reached,uint64_t peak,uint32_t peakTick,double elapsed,const uint8_t*state,uint32_t preview) {
  printf("{\"kind\":\"progress\",\"tick\":%u,\"susceptible\":%llu,\"believers\":%llu,\"factCheckers\":%llu,\"reached\":%llu,\"maxBelievers\":%llu,\"maxBelieversTick\":%u,\"elapsedSeconds\":%.6f",tick,(unsigned long long)s,(unsigned long long)b,(unsigned long long)f,(unsigned long long)reached,(unsigned long long)peak,peakTick,elapsed);
  if(state) {
    fputs(",\"nodeStates\":\"",stdout);
    std::vector<char> text(preview);
    for(uint32_t i=0;i<preview;i++)text[i]='0'+state[i];
    fwrite(text.data(),1,text.size(),stdout);putchar('"');
  }
  puts("}");fflush(stdout);
}

int main(int argc,char**argv) {
  MPI_Init(&argc,&argv);
  int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
  Config c=args(argc,argv);
  if(c.graphOnly) {
    if(rank==0) {
      Graph graph=make_graph(c.n,c.degree,c.seed);
      if(c.graphOnly==2) {
        // Packed little-endian uint32 endpoints; avoid millions of printf calls
        // and JSON conversions on the native/Python/browser hot path.
        #ifdef _WIN32
        _setmode(1,_O_BINARY); // stdout file descriptor
        #endif
        uint32_t header[4]={0x43464253,c.n,(uint32_t)graph.edges,(uint32_t)(graph.edges>>32)};
        fwrite(header,sizeof(uint32_t),4,stdout);
        uint32_t block[16384];size_t used=0;
        for(uint32_t node=0;node<c.n;node++)for(uint64_t p=graph.offset[node];p<graph.offset[node+1];p++) {
          uint32_t neighbour=graph.adj[p];
          if(node<neighbour) {
            block[used++]=node;block[used++]=neighbour;
            if(used==16384){fwrite(block,sizeof(uint32_t),used,stdout);used=0;}
          }
        }
        if(used)fwrite(block,sizeof(uint32_t),used,stdout);
        fflush(stdout);free(graph.offset);free(graph.adj);
      } else {
      printf("{\"totalNodes\":%u,\"edgeCount\":%llu,\"edges\":[",c.n,(unsigned long long)graph.edges);
      int comma=0;
      for(uint32_t node=0;node<c.n;node++)for(uint64_t p=graph.offset[node];p<graph.offset[node+1];p++) {
        uint32_t neighbor=graph.adj[p];
        if(node<neighbor){printf("%s%u,%u",comma?",":"",node,neighbor);comma=1;}
      }
      puts("]}");free(graph.offset);free(graph.adj);
      }
    }
    MPI_Finalize();return 0;
  }
  if((uint32_t)size>c.n){fprintf(stderr,"MPI ranks cannot exceed nodes\n");MPI_Abort(MPI_COMM_WORLD,4);}
  int devices=0;CUDA_OK(cudaGetDeviceCount(&devices));
  if(devices<1){fprintf(stderr,"No CUDA GPU visible\n");MPI_Abort(MPI_COMM_WORLD,5);}
  // Local rank makes device placement correct even across multiple hosts.
  MPI_Comm shared;MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&shared);
  int localRank;MPI_Comm_rank(shared,&localRank);MPI_Comm_free(&shared);
  int device=localRank%devices;CUDA_OK(cudaSetDevice(device));
  MPI_Barrier(MPI_COMM_WORLD);
  double total0=MPI_Wtime(),graph0=total0,communicationSecs=0;
  uint64_t edges=0;
  Graph graph={0,0,0};
  if(rank==0){graph=make_graph(c.n,c.degree,c.seed);edges=graph.edges;}
  MPI_Bcast(&edges,1,MPI_UINT64_T,0,MPI_COMM_WORLD);
  std::vector<int> counts(size),displacements(size);
  for(int r=0;r<size;r++) {
    displacements[r]=partition_begin(c.n,size,r);
    counts[r]=partition_begin(c.n,size,r+1)-displacements[r];
  }
  uint32_t begin=displacements[rank],count=counts[rank];
  std::vector<uint64_t> localOffsets(count+1);
  std::vector<uint32_t> globalNeighbors;
  // Root sends each rank only the adjacency rows it owns, retaining all remote IDs.
  if(rank==0) {
    for(int r=0;r<size;r++) {
      uint32_t start=displacements[r],length=counts[r];
      uint64_t base=graph.offset[start],entries=graph.offset[start+length]-base;
      if(entries>INT_MAX){fprintf(stderr,"Partition exceeds MPI count limit\n");MPI_Abort(MPI_COMM_WORLD,4);}
      std::vector<uint64_t> offsets(length+1);
      for(uint32_t i=0;i<=length;i++)offsets[i]=graph.offset[start+i]-base;
      if(r==0) {
        localOffsets=std::move(offsets);
        globalNeighbors.assign(graph.adj+base,graph.adj+base+entries);
      } else {
        MPI_Send(offsets.data(),length+1,MPI_UINT64_T,r,10,MPI_COMM_WORLD);
        MPI_Send(graph.adj+base,(int)entries,MPI_UINT32_T,r,11,MPI_COMM_WORLD);
      }
    }
    free(graph.offset);free(graph.adj);
  } else {
    MPI_Recv(localOffsets.data(),count+1,MPI_UINT64_T,0,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    globalNeighbors.resize(localOffsets.back());
    MPI_Recv(globalNeighbors.data(),(int)globalNeighbors.size(),MPI_UINT32_T,0,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
  Partition part=make_partition(c.n,size,rank,std::move(localOffsets),std::move(globalNeighbors));
  std::vector<int> sendCounts(size),sendDisplacements(size);
  // One-time exchange of which boundary-node states each other rank needs.
  MPI_Alltoall(part.requestCounts.data(),1,MPI_INT,sendCounts.data(),1,MPI_INT,MPI_COMM_WORLD);
  for(int r=1;r<size;r++)sendDisplacements[r]=sendDisplacements[r-1]+sendCounts[r-1];
  int sendTotal=sendDisplacements.back()+sendCounts.back();
  std::vector<uint32_t> requestedIds(sendTotal);
  MPI_Alltoallv(part.ghosts.data(),part.requestCounts.data(),part.requestDisplacements.data(),MPI_UINT32_T,
    requestedIds.data(),sendCounts.data(),sendDisplacements.data(),MPI_UINT32_T,MPI_COMM_WORLD);
  for(uint32_t id:requestedIds)if(id<begin||id>=begin+count){fprintf(stderr,"Invalid ghost owner\n");MPI_Abort(MPI_COMM_WORLD,6);}
  std::vector<uint8_t> state(count+part.ghosts.size(),S),sendState(sendTotal),seen(count,0);
  if(c.source>=begin&&c.source<begin+count){state[c.source-begin]=B;seen[c.source-begin]=1;}
  auto exchange=[&]() {
    double start=MPI_Wtime();
    for(int i=0;i<sendTotal;i++)sendState[i]=state[requestedIds[i]-begin];
    // Host buffers work with standard Colab Open MPI; CUDA-aware MPI is not needed.
    MPI_Alltoallv(sendState.data(),sendCounts.data(),sendDisplacements.data(),MPI_UNSIGNED_CHAR,
      state.data()+count,part.requestCounts.data(),part.requestDisplacements.data(),MPI_UNSIGNED_CHAR,MPI_COMM_WORLD);
    communicationSecs+=MPI_Wtime()-start;
  };
  exchange();
  uint64_t crossEntries=0;
  MPI_Reduce(&part.crossEntries,&crossEntries,1,MPI_UINT64_T,MPI_SUM,0,MPI_COMM_WORLD);
  uint64_t localMeta[5]={begin,begin+count,part.ghosts.size(),part.adjacency.size(),(uint64_t)device};
  std::vector<uint64_t> metadata(rank==0?size*5:0);
  MPI_Gather(localMeta,5,MPI_UINT64_T,metadata.data(),5,MPI_UINT64_T,0,MPI_COMM_WORLD);
  if(rank==0) {
    printf("{\"kind\":\"partition\",\"mpiRanks\":%d,\"crossEdges\":%llu,\"partitions\":[",size,(unsigned long long)(crossEntries/2));
    for(int r=0;r<size;r++) {
      uint64_t* m=metadata.data()+r*5;
      printf("%s{\"rank\":%d,\"begin\":%llu,\"end\":%llu,\"ghostNodes\":%llu,\"adjacencyEntries\":%llu,\"gpu\":%llu}",r?",":"",r,
        (unsigned long long)m[0],(unsigned long long)m[1],(unsigned long long)m[2],(unsigned long long)m[3],(unsigned long long)m[4]);
    }
    puts("]}");fflush(stdout);
  }
  double graphSecs=MPI_Wtime()-graph0;
  uint64_t *doff;uint32_t*dadj;uint8_t*dcur,*dnext;
  CUDA_OK(cudaMalloc((void**)&doff,part.offset.size()*sizeof(uint64_t)));
  CUDA_OK(cudaMalloc((void**)&dadj,std::max((size_t)1,part.adjacency.size())*sizeof(uint32_t)));
  CUDA_OK(cudaMalloc((void**)&dcur,state.size()));CUDA_OK(cudaMalloc((void**)&dnext,count));
  CUDA_OK(cudaMemcpy(doff,part.offset.data(),part.offset.size()*sizeof(uint64_t),cudaMemcpyHostToDevice));
  if(!part.adjacency.empty())CUDA_OK(cudaMemcpy(dadj,part.adjacency.data(),part.adjacency.size()*sizeof(uint32_t),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dcur,state.data(),state.size(),cudaMemcpyHostToDevice));
  std::vector<uint8_t> fullState(rank==0?c.n:0);
  auto gather=[&]() {
    double start=MPI_Wtime();
    MPI_Gatherv(state.data(),count,MPI_UNSIGNED_CHAR,fullState.data(),counts.data(),displacements.data(),MPI_UNSIGNED_CHAR,0,MPI_COMM_WORLD);
    communicationSecs+=MPI_Wtime()-start;
  };
  gather();
  uint64_t reached=1,peak=1,finalS=c.n-1,finalB=1,finalF=0;
  uint32_t peakTick=0;
  if(rank==0)emit(0,finalS,finalB,finalF,reached,peak,peakTick,0,fullState.data(),c.n);
  MPI_Barrier(MPI_COMM_WORLD);double sim0=MPI_Wtime(),lastFrame=sim0;
  for(uint32_t tick=1;tick<=c.ticks;tick++) {
    step_kernel<<<(count+255)/256,256>>>(doff,dadj,dcur,dnext,begin,count,tick,c.seed,c.alpha,c.beta,c.verify,c.forget);
    CUDA_OK(cudaGetLastError());CUDA_OK(cudaMemcpy(state.data(),dnext,count,cudaMemcpyDeviceToHost));
    uint64_t lc[4]={0,0,0,0},gc[4]={0,0,0,0};
    for(uint32_t i=0;i<count;i++){lc[state[i]]++;if(state[i]!=S&&!seen[i]){seen[i]=1;lc[3]++;}}
    double reduce0=MPI_Wtime();MPI_Reduce(lc,gc,4,MPI_UINT64_T,MPI_SUM,0,MPI_COMM_WORLD);
    communicationSecs+=MPI_Wtime()-reduce0;
    if(rank==0) {
      reached+=gc[3];finalS=gc[S];finalB=gc[B];finalF=gc[F];
      if(finalB>peak){peak=finalB;peakTick=tick;}
    }
    // Synchronous ticks: remote states for t+1 come from the completed tick t.
    if(tick<c.ticks){exchange();CUDA_OK(cudaMemcpy(dcur,state.data(),state.size(),cudaMemcpyHostToDevice));}
    if(tick%c.sample==0||tick==c.ticks) {
      // Counts keep the requested tick sampling. Large visual snapshots are
      // wall-clock limited; all ranks agree before entering the gather.
      int snapshot=0;
      if(rank==0)snapshot=c.n<100000||tick==c.ticks||MPI_Wtime()-lastFrame>=0.25;
      double frameDecision=MPI_Wtime();MPI_Bcast(&snapshot,1,MPI_INT,0,MPI_COMM_WORLD);
      communicationSecs+=MPI_Wtime()-frameDecision;
      if(snapshot){gather();lastFrame=MPI_Wtime();}
      if(rank==0)emit(tick,finalS,finalB,finalF,reached,peak,peakTick,MPI_Wtime()-sim0,snapshot?fullState.data():nullptr,c.n);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);double simSecs=MPI_Wtime()-sim0,totalSecs=MPI_Wtime()-total0;
  double maxCommunication=0;
  MPI_Reduce(&communicationSecs,&maxCommunication,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  if(rank==0) {
    printf("{\"kind\":\"summary\",\"mode\":\"hybrid\",\"nodes\":%u,\"edges\":%llu,\"ticks\":%u,\"susceptible\":%llu,\"believers\":%llu,\"factCheckers\":%llu,\"reached\":%llu,\"maxBelievers\":%llu,\"maxBelieversTick\":%u,\"graphSeconds\":%.6f,\"simulationSeconds\":%.6f,\"totalSeconds\":%.6f,\"communicationSeconds\":%.6f,\"crossEdges\":%llu,\"mpiRanks\":%d,\"gpuCount\":%d,\"sourceNode\":%u,\"seed\":%llu,\"meanDegree\":%u}\n",
      c.n,(unsigned long long)edges,c.ticks,(unsigned long long)finalS,(unsigned long long)finalB,(unsigned long long)finalF,
      (unsigned long long)reached,(unsigned long long)peak,peakTick,graphSecs,simSecs,totalSecs,maxCommunication,
      (unsigned long long)(crossEntries/2),size,devices,c.source,(unsigned long long)c.seed,c.degree);fflush(stdout);
  }
  cudaFree(doff);cudaFree(dadj);cudaFree(dcur);cudaFree(dnext);
  MPI_Finalize();return 0;
}
