#pragma once
#include <stdint.h>
#include <algorithm>
#include <utility>
#include <vector>

#ifdef __CUDACC__
#define SBFC_HD __host__ __device__
#else
#define SBFC_HD
#endif

enum { S=0, B=1, F=2 };

SBFC_HD static uint64_t mix64(uint64_t x) {
    x+=0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}

SBFC_HD static double rnd(uint64_t seed,uint32_t tick,uint32_t node,uint32_t stream) {
    uint64_t k=seed^((uint64_t)tick*0xd6e8feb86659fd93ULL)^((uint64_t)node*0xa5a3564e27f8862bULL)^((uint64_t)stream*0x9e3779b97f4a7c15ULL);
    return (double)(mix64(k)>>11)*(1.0/9007199254740992.0);
}

// Uses compact local/ghost indices for neighbours, but global IDs for randomness.
// The same function is exercised by CPU partition-equivalence tests and CUDA.
SBFC_HD static uint8_t update_state(const uint64_t* off,const uint32_t* adj,
    const uint8_t* cur,uint32_t local,uint32_t global,uint32_t tick,uint64_t seed,
    double alpha,double beta,double verify,double forget) {
    uint8_t state=cur[local],out=state;
    if(state==S) {
        uint32_t nb=0,nf=0;
        for(uint64_t p=off[local];p<off[local+1];p++) {
            uint8_t q=cur[adj[p]]; nb+=q==B; nf+=q==F;
        }
        double wb=nb*(1.0+alpha),wf=nf*(1.0-alpha),den=wb+wf;
        if(den>0) {
            double r=rnd(seed,tick,global,0),pb=beta*wb/den,pf=beta*wf/den;
            if(r<pb)out=B;else if(r<pb+pf)out=F;
        }
    } else if(state==B) {
        if(rnd(seed,tick,global,0)<forget)out=S;
        else if(rnd(seed,tick,global,1)<verify)out=F;
    } else if(rnd(seed,tick,global,0)<forget)out=S;
    return out;
}

static uint32_t partition_begin(uint32_t n,int ranks,int rank) {
    return (uint64_t)n*rank/ranks;
}

static int partition_owner(uint32_t node,uint32_t n,int ranks) {
    return (int)(((uint64_t)(node+1)*ranks-1)/n);
}

struct Partition {
    uint32_t begin,count;
    std::vector<uint64_t> offset;
    std::vector<uint32_t> adjacency,ghosts;
    std::vector<int> requestCounts,requestDisplacements;
    uint64_t crossEntries=0;
};

// Every original adjacency entry is retained. Only its state-buffer index changes.
static Partition make_partition(uint32_t n,int ranks,int rank,
    std::vector<uint64_t> offset,std::vector<uint32_t> neighbors) {
    Partition p;
    p.begin=partition_begin(n,ranks,rank);
    p.count=partition_begin(n,ranks,rank+1)-p.begin;
    p.offset=std::move(offset);
    p.requestCounts.assign(ranks,0);p.requestDisplacements.assign(ranks,0);
    for(uint32_t node:neighbors)if(node<p.begin||node>=p.begin+p.count) {
        p.ghosts.push_back(node);p.crossEntries++;
    }
    std::sort(p.ghosts.begin(),p.ghosts.end());
    p.ghosts.erase(std::unique(p.ghosts.begin(),p.ghosts.end()),p.ghosts.end());
    for(uint32_t node:p.ghosts)p.requestCounts[partition_owner(node,n,ranks)]++;
    for(int r=1;r<ranks;r++)p.requestDisplacements[r]=p.requestDisplacements[r-1]+p.requestCounts[r-1];
    p.adjacency.reserve(neighbors.size());
    for(uint32_t node:neighbors) {
        if(node>=p.begin&&node<p.begin+p.count)p.adjacency.push_back(node-p.begin);
        else p.adjacency.push_back(p.count+(uint32_t)(std::lower_bound(p.ghosts.begin(),p.ghosts.end(),node)-p.ghosts.begin()));
    }
    return p;
}
