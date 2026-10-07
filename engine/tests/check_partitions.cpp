// CPU tests of the exact partition remapping and state-update function used by CUDA.
// g++ -O2 -std=c++14 engine/tests/check_partitions.cpp -o check-partitions
#include "../src/sbfc_core.hpp"
#include <cassert>
#include <cstdio>

static void check(uint32_t n,int ranks,int ticks) {
    // Ring + hub + chords exercise local and remote neighbours, including hubs
    // requested by multiple ranks and uneven ownership boundaries.
    std::vector<std::vector<uint32_t>> rows(n);
    auto edge=[&](uint32_t a,uint32_t b) {
        if(a==b)return;
        rows[a].push_back(b);rows[b].push_back(a);
    };
    for(uint32_t i=0;i<n;i++){edge(i,(i+1)%n);if(i>1)edge(0,i);edge(i,(i+7)%n);}
    std::vector<uint64_t> off(n+1,0);std::vector<uint32_t> adj;
    for(uint32_t i=0;i<n;i++){adj.insert(adj.end(),rows[i].begin(),rows[i].end());off[i+1]=adj.size();}
    std::vector<Partition> parts;
    uint64_t totalEntries=0,crossEntries=0;
    for(int rank=0;rank<ranks;rank++) {
        uint32_t begin=partition_begin(n,ranks,rank),end=partition_begin(n,ranks,rank+1);
        std::vector<uint64_t> localOff(end-begin+1);
        for(uint32_t i=begin;i<=end;i++)localOff[i-begin]=off[i]-off[begin];
        std::vector<uint32_t> neighbors(adj.begin()+off[begin],adj.begin()+off[end]);
        Partition p=make_partition(n,ranks,rank,std::move(localOff),neighbors);
        assert(p.adjacency.size()==neighbors.size());
        for(size_t i=0;i<neighbors.size();i++) {
            uint32_t mapped=p.adjacency[i];
            uint32_t restored=mapped<p.count?p.begin+mapped:p.ghosts[mapped-p.count];
            assert(restored==neighbors[i]);
        }
        for(int owner=0;owner<ranks;owner++) {
            for(int j=0;j<p.requestCounts[owner];j++) {
                uint32_t id=p.ghosts[p.requestDisplacements[owner]+j];
                assert(partition_owner(id,n,ranks)==owner);
            }
        }
        totalEntries+=p.adjacency.size();crossEntries+=p.crossEntries;
        parts.push_back(std::move(p));
    }
    assert(totalEntries==adj.size());
    uint64_t expectedCross=0;
    for(uint32_t i=0;i<n;i++)for(uint64_t j=off[i];j<off[i+1];j++)
        expectedCross+=partition_owner(i,n,ranks)!=partition_owner(adj[j],n,ranks);
    assert(crossEntries==expectedCross);

    std::vector<uint8_t> reference(n,S),distributed(n,S),nextReference(n),nextDistributed(n);
    reference[n-1]=distributed[n-1]=B;
    reference[n/2]=distributed[n/2]=F;
    for(int tick=1;tick<=ticks;tick++) {
        for(uint32_t i=0;i<n;i++)nextReference[i]=update_state(off.data(),adj.data(),reference.data(),i,i,tick,42,.3,.5,.05,.1);
        // Pack the one-time request plan in exactly the same owner/requester
        // ordering as Alltoallv; no global state is used directly by update_state.
        std::vector<std::vector<uint8_t>> ghostBuffers(ranks);
        for(int requester=0;requester<ranks;requester++) {
            const auto& p=parts[requester];ghostBuffers[requester].resize(p.ghosts.size());
            for(int owner=0;owner<ranks;owner++)for(int k=0;k<p.requestCounts[owner];k++) {
                int index=p.requestDisplacements[owner]+k;
                uint32_t id=p.ghosts[index];
                assert(id>=parts[owner].begin&&id<parts[owner].begin+parts[owner].count);
                ghostBuffers[requester][index]=distributed[id];
            }
        }
        for(int rank=0;rank<ranks;rank++) {
            const auto& p=parts[rank];
            std::vector<uint8_t> state(distributed.begin()+p.begin,distributed.begin()+p.begin+p.count);
            state.insert(state.end(),ghostBuffers[rank].begin(),ghostBuffers[rank].end());
            for(uint32_t local=0;local<p.count;local++)nextDistributed[p.begin+local]=update_state(
                p.offset.data(),p.adjacency.data(),state.data(),local,p.begin+local,tick,42,.3,.5,.05,.1);
        }
        assert(nextReference==nextDistributed);
        reference.swap(nextReference);distributed.swap(nextDistributed);
    }
    std::printf("PASS: %u nodes, %d ranks, %d ticks; all edges retained and distributed states equal reference\n",n,ranks,ticks);
}

int main() {
    for(int ranks: {1,2,3,4,8})check(17,ranks,40);
    for(int ranks: {2,4,8})check(1000,ranks,40);
    check(100000,4,5);check(100000,8,5);
}
