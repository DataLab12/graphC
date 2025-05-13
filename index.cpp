
#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <string>

#include <cstdio>
#include <climits>
#include <algorithm>
#include <set>
#include <map>
#include <vector>
#include <map>
#include <sys/time.h>
#include <iostream>
#include <list>
#include "GraphBplus_Harary.cpp"


static Graph readGraph(const char* const name,std::map<int,int>& labels)
{
  // read input from file
  FILE* fin = fopen(name, "rt");
  if (fin == NULL) {printf("ERROR: could not open input file %s\n", name); exit(-1);}
  size_t linesize = 256;
  char buf[linesize];
  char* ptr = buf;
  getline(&ptr, &linesize, fin);  // skip first line

  int selfedges = 0, wrongweights = 0, duplicates = 0, inconsistent = 0, line = 1, cnt = 0;
  int src, dst, wei;
  std::map<int, int> map;  // map node IDs to contiguous IDs
  std::set<std::pair<int, int>> set2;
  std::set<std::tuple<int, int, int>> set3;
  while (fscanf(fin, "%d,%d,%d", &src, &dst, &wei) == 3) {
    if(src==dst){
      selfedges++;
    }
    if(false){
      selfedges++;
    }
     else if ((wei < -1) || (wei > 1)) {
      wrongweights++;
    } else if (set2.find(std::make_pair(std::min(src, dst), std::max(src, dst))) != set2.end()) {
      if (set3.find(std::make_tuple(std::min(src, dst), std::max(src, dst), wei)) != set3.end()) {
        duplicates++;
      } else {
        inconsistent++;
      }
    } else {
      set2.insert(std::make_pair(std::min(src, dst), std::max(src, dst)));
      set3.insert(std::make_tuple(std::min(src, dst), std::max(src, dst), wei));
      if (map.find(src) == map.end()) {
        map[src] = cnt++;
      }
      if (map.find(dst) == map.end()) {
        map[dst] = cnt++;
      }
    }
    line++;
  }
  fclose(fin);

 printf("  read %d lines\n", line);
  if (selfedges > 0) printf("  skipped %d self-edges\n", selfedges);
  if (wrongweights > 0) printf("  skipped %d edges with out-of-range weights\n", wrongweights);
  if (duplicates > 0) printf("  skipped %d duplicate edges\n", duplicates);
  if (inconsistent > 0) printf("  skipped %d inconsistent edges\n", inconsistent);

  // compute CCs with union find
  int* const label = new int [cnt];
  for (int v = 0; v < cnt; v++) {
    label[v] = v;
  }
  for (auto ele: set3) {
    const int src = map[std::get<0>(ele)];
    const int dst = map[std::get<1>(ele)];
    const int vstat = representative(src, label);
    const int ostat = representative(dst, label);
    if (vstat != ostat) {
      if (vstat < ostat) {
        label[ostat] = vstat;
      } else {
        label[vstat] = ostat;
      }
    }
  }
  for (int v = 0; v < cnt; v++) {
    int next, vstat = label[v];
    while (vstat > (next = label[vstat])) {
      vstat = next;
    }
    label[v] = vstat;
  }

  // determine CC sizes
  int* const size = new int [cnt];
  for (int v = 0; v < cnt; v++) {
    size[v] = 0;
  }
  for (int v = 0; v < cnt; v++) {
    size[label[v]]++;
  }

  // find largest CC
  int hi = 0;
  for (int v = 1; v < cnt; v++) {
    if (size[hi] < size[v]) hi = v;
  }

  Graph g;
  g.origID = new int [cnt];  // upper bound on size
  int nodes = 0, edges = 0;
  std::map<int, int> newmap;  // map node IDs to contiguous IDs
  std::set<std::pair<int, int>>* const node = new std::set<std::pair<int, int>> [cnt];  // upper bound on size
  for (auto ele: set3) {
    const int src = std::get<0>(ele);
    const int dst = std::get<1>(ele);
    const int wei = std::get<2>(ele);
      if (newmap.find(src) == newmap.end()) {
        g.origID[nodes] = src;
        labels[src]=label[map[src]];
        newmap[src] = nodes++;
      }
      if (newmap.find(dst) == newmap.end()) {
        g.origID[nodes] = dst;
        labels[dst]=label[map[dst]];
        newmap[dst] = nodes++;

      }
      if(src!=dst){
      node[newmap[src]].insert(std::make_pair(newmap[dst], wei));
      node[newmap[dst]].insert(std::make_pair(newmap[src], wei));
      edges += 2;
      }
  }

  // create graph in CSR format
  g.nodes = nodes;
  g.edges = edges;
  g.nindex = new int [g.nodes + 1];
  g.nlist = new int [g.edges];
  g.eweight = new int [g.edges];
  int acc = 0;
  for (int v = 0; v < g.nodes; v++) {
    g.nindex[v] = acc;
    for (auto ele: node[v]) {
      const int dst = ele.first;
      const int wei = ele.second;
      g.nlist[acc] = dst;
      g.eweight[acc] = wei;
      acc++;
    
    }
  }
  g.nindex[g.nodes] = acc;

  delete [] node;

  return g;
}

struct CPUTimer
{
  timeval beg, end;
  CPUTimer() {}
  ~CPUTimer() {}
  void start() {gettimeofday(&beg, NULL);}
  double elapsed() {gettimeofday(&end, NULL); return end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;}
};
std::vector<double> evaluate(Graph g_temp,std::map<int,int> labels){
std::vector<double> value;
double pos_within=0,pos_between=0,neg_within=0,neg_between =0;
  for (int v = 0; v < g_temp.nodes; v++) {
    int index=g_temp.nindex[v];
    int index_max=g_temp.nindex[v+1];
    while(index<index_max){
      if(v<g_temp.nlist[index]){
      if(g_temp.eweight[index]==-1){
          if(labels[g_temp.origID[v]]==labels[g_temp.origID[g_temp.nlist[index]]]){
          neg_within=neg_within+1;
        }
        else{
          neg_between=neg_between+1;
        }
      }else{
           if(labels[g_temp.origID[v]]==labels[g_temp.origID[g_temp.nlist[index]]]){
          pos_within=pos_within+1;
        }
        else{
          pos_between=pos_between+1;
        }
      }
      }
    index=index+1;
    }
  }
  value.push_back((pos_within/(pos_between+pos_within)));
  value.push_back((neg_between/(neg_between+neg_within)));
    value.push_back((pos_between/(pos_between+pos_within)));
  value.push_back((neg_within/(neg_between+neg_within)));
   value.push_back((pos_between/(pos_between+pos_within))+(neg_within/(neg_within+neg_between)));
   value.push_back(((pos_between+neg_within)/(g_temp.edges/2))*100);
return value;
}

int main(int argc, char* argv[])
{
  printf("Harary Clustering based on graphB++ balancing code for signed network graphs (%s)\n", __FILE__);
  printf("Copyright 2024 Texas State University\n");
  CPUTimer overall;
  overall.start();
  // process command line and read input
  if (argc != 8) {printf("USAGE: %s input_file_name iteration_count alpha beta epsilon time_limit CC_size_limit\n", argv[0]); exit(-1);}
  printf("input: %s\n", argv[1]);
std::set<int> R;
const char* name=argv[1];
int i=atoi(argv[2]);
double alpha=atof(argv[3]);
double beta=atof(argv[4]);
double epsilon=atof(argv[5]);
double time_limit=atof(argv[6]);
int gamma=atof(argv[7]);
std::map<int,int> labels;
Graph g_temp=readGraph(name,labels);
//std::map<int,int> degree;
std::cout<<"number of nodes: "<<g_temp.nodes<<std::endl;
std::cout<<"number of edges: "<<g_temp.edges/2<<std::endl;
int number_of_clusters=1;
int label_counter=0;
double unhappy_ratio_new=1;
double unhappy_ratio_prev=__DBL_MAX__;
std::map<int,int> mapping;
std::set<int> stop_labels;
std::vector<double> pos_in;
std::vector<double> neg_out;
std::vector<double> pos_out;
std::vector<double> neg_in;
std::vector<double> unhappy;

    for (const auto &v : labels){
      if (mapping.find(v.second) == mapping.end()) {
    mapping[v.second]=label_counter;
    label_counter++;
      }
    }
    for (const auto &v : labels){
      labels[v.first]=mapping[v.second];
    }
    int splits=0;
while(true){
  if(time_limit!=-1.0){
  if(overall.elapsed()>=time_limit){
    break;
  }
  }
  int number_of_nodes=g_temp.nodes;
  int label_max=-1;
  bool terminate=true;
  std::map<int,int> counter;
  for (const auto &v : labels){
    counter[v.second]=counter[v.second]+1;
  }
    for (const auto &v : labels){
      if(counter[v.second]<=gamma){
        continue;
      }
      if(stop_labels.find(v.second)==stop_labels.end()){
        label_max=v.second;
        terminate=false;
        break;
      }
    }
    if(terminate){
      break;
    }
    for (const auto &v : labels){
            if(v.second!=label_max){
          R.insert(v.first);
          number_of_nodes=number_of_nodes-1;
            }
    }

  std::vector<std::vector<int>> harary_cuts=GraphBplus(name,i,R,alpha,beta);

  std::map<int,int> temp_labels;
  temp_labels.insert(labels.begin(), labels.end());
  int label_temp=label_counter;
 for(int i=0;i<number_of_nodes;i++){

   if(harary_cuts[i].size()==0){
     continue;
   }
        for(int j=0;j<harary_cuts[i].size();j++){
          labels[harary_cuts[i][j]]=label_counter;
        }
        label_counter++;
        }

R.clear();

  double unhappy_ratio_new_temp=unhappy_ratio_new;
  double unhappy_ratio_prev_temp=unhappy_ratio_prev;

  double temp=unhappy_ratio_new;
  std::vector<double> value=evaluate(g_temp,labels);
  unhappy_ratio_new=value[4];
  unhappy_ratio_prev=temp;
  if(unhappy_ratio_prev-unhappy_ratio_new<epsilon){
    labels.clear();
  labels.insert(temp_labels.begin(), temp_labels.end());
  label_counter=label_temp;
  stop_labels.insert(label_max);
  unhappy_ratio_new=unhappy_ratio_new_temp;
  unhappy_ratio_prev=unhappy_ratio_prev_temp;
  //std::cout<<"No improvement anymore for CC of label " <<label_max<<std::endl;

  }else{
    splits++;
  pos_in.push_back(value[0]);
  neg_out.push_back(value[1]);
  pos_out.push_back(value[2]);
  neg_in.push_back(value[3]);
  unhappy.push_back(value[5]);
  std::cout<<"Improvement after split for CC of label "<<label_max<<": "<<unhappy_ratio_prev-unhappy_ratio_new<<std::endl;
  }
}

double pos_within=0,pos_between=0,neg_within=0,neg_between =0;
  for (int v = 0; v < g_temp.nodes; v++) {
    int index=g_temp.nindex[v];
    int index_max=g_temp.nindex[v+1];
    while(index<index_max){
      if(v<g_temp.nlist[index]){
      if(g_temp.eweight[index]==-1){
          if(labels[g_temp.origID[v]]==labels[g_temp.origID[g_temp.nlist[index]]]){
          neg_within=neg_within+1;
        }
        else{
          neg_between=neg_between+1;
        }
      }else{
           if(labels[g_temp.origID[v]]==labels[g_temp.origID[g_temp.nlist[index]]]){
          pos_within=pos_within+1;
        }
        else{
          pos_between=pos_between+1;
        }
      }
      }
    index=index+1;
    }
  }
  double unhappy_ratio=(pos_between+neg_within)/(g_temp.edges/2);

  std::map<int,int> counter;
for (const auto &v : labels){
  counter[v.second]=counter[v.second]+1;
    }

  int num_isolated_communities=0;
for (const auto &v : counter){
  if(v.second==1){
    num_isolated_communities++;
  }
    }
std::cout<<"pos_in: "<<(pos_within/(pos_between+pos_within))<<std::endl;
std::cout<<"neg_out: "<<(neg_between/(neg_between+neg_within))<<std::endl;
std::cout<<"pos_within: "<<pos_within<<std::endl;
std::cout<<"pos_between: "<<pos_between<<std::endl;
std::cout<<"neg_within: "<<neg_within<<std::endl;
std::cout<<"neg_between: "<<neg_between<<std::endl;

std::cout<<"happy_ratio: "<<(pos_within+neg_between)*100/(g_temp.edges/2)<<std::endl;
std::cout<<"unhappy_ratio: "<<unhappy_ratio*100<<std::endl;
std::cout<<"optimal number of clusters: "<<counter.size()<<std::endl;
std::cout<<"number of splitting operations: "<<splits<<std::endl;
std::cout<<"number of communities without isolated nodes: "<<counter.size()-num_isolated_communities<<std::endl;

// output results to file
  std::string name_edge(name);
  name_edge.pop_back();
  name_edge.pop_back();
  name_edge.pop_back();
  name_edge.pop_back();
  name_edge.append("_labels.txt");
  FILE *labels_file = fopen(name_edge.c_str(), "wt");

  fprintf(labels_file, "original node ID, label\n");
  for (const auto &v : labels){
    fprintf(labels_file, "%d,%d\n",v.first, v.second);
  }
  fclose(labels_file);
std::cout<<"Outputted labels to "<<name_edge<<std::endl;

// output results to file
/*  std::string name_edge1(name);
  name_edge1.pop_back();
  name_edge1.pop_back();
  name_edge1.pop_back();
  name_edge1.pop_back();
  name_edge1.append("_outliers.txt");
  FILE *outliers_file = fopen(name_edge1.c_str(), "wt");

  fprintf(outliers_file, "original node ID\n");
  for (const auto &v : labels){
    if(counter[v.second]==1){
    fprintf(outliers_file, "%d\n",v.first);
    }
  }
  fclose(outliers_file);
std::cout<<"Outputted outliers to "<<name_edge1<<std::endl;*/

// output results to file
  std::string name_edge1(name);
  name_edge1.pop_back();
  name_edge1.pop_back();
  name_edge1.pop_back();
  name_edge1.pop_back();
  name_edge1.append("_posin_negout.txt");
  FILE *improv_file = fopen(name_edge1.c_str(), "wt");

  fprintf(improv_file, "pos_in,neg_out,pos_out,neg_in,unhappy\n");
  for(int i=0;i<pos_in.size();i++){
    fprintf(improv_file, "%f,%f,%f,%f,%f\n",pos_in[i],neg_out[i],pos_out[i],neg_in[i],unhappy[i]);

  }
  fclose(improv_file);
std::cout<<"Outputted pos_in and neg_out to "<<name_edge1<<std::endl;

  freeGraph(g_temp);
  printf("overall runtime with I/O: %.6f s\n", overall.elapsed());
}
