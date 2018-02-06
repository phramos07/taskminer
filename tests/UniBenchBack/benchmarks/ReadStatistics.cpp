#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cstdio>

using namespace std;

#define NTargets 6

std::string getName (char *argv[]) {
  std::string tmp;
  int i = 0;
  for (int i = 0; argv[1][i] != '\0'; i++) 
    tmp += argv[1][i];
  return tmp;
}

std::string extract (std::string str, std::string input) {
  int pos = input.find(str);
  std::string tmp;
  if (str != "Writing output to file") {
    for (int i = 0; i < pos; i++)
      tmp += input[i];
    return tmp;
  }
  pos = input.rfind("/");
  for (int i = pos + 1; input[i] != '.'; i++)
    tmp += input[i];
  return tmp;
}

void createCSV (vector <vector <string> > & data) {
  ofstream file ("benchmarksStatistics.csv");
  if (!file.is_open()) {
    printf("Error to create the benchmarksStatistics.csv file.\n");
    return;
  }
  for (int i = 0, ie = data.size(); i != ie; i++) {
    if (data[i].size() == 1)
      continue;
    for (int j = 0, je = data[i].size() - 1; j != je; j++)
      file << data[i][j] << ",";
    if (data[i].size() != NTargets)
      for (int j = data[i].size(); j < NTargets; j++)
        file << "0,";
    file << data[i][(data[i].size() - 1)] << ","; 
    file << "\n";
  }
  printf("Wrote!!!\n");
  file.close();
}

void readFile (std::string name) {
  ifstream file(name);
  if (!file.is_open()) {
    printf("Error to try open the file\n");
    return;
  }
  string line;
  vector <vector <string> > found;
  vector <string> targets;
  vector <string> tmp; 
  // The headers to find.
  targets.push_back("Writing output to file");
  targets.push_back("PTRRangeAnalysis - Number of memory access");
  targets.push_back("PTRRangeAnalysis - Number of memory analyzed access");
  targets.push_back("writeExpressions - Number of analyzable loops");
  targets.push_back("writeExpressions - Number of annotated loops");
  targets.push_back("writeExpressions - Number of loops");
  // The headers to write.
  tmp.push_back("Name");
  tmp.push_back("Memory Access");
  tmp.push_back("Analyzed Memory Access");
  tmp.push_back("Analyzed Loops");
  tmp.push_back("Annotated Loops");
  tmp.push_back("Loops"); 
  found.push_back(tmp);
  tmp.erase(tmp.begin(), tmp.end());
  
  std::string lastName;
  while (getline(file,line)) {
    if (line.find(targets[0]) != string::npos)
      lastName = extract(targets[0],line);
    
    if (tmp.size() > 0 && tmp[0] != lastName) {
      found.push_back(tmp);
      tmp.erase(tmp.begin(), tmp.end()); 
    }
    
    for (int i = 0; i < targets.size(); i++)
      if (line.find(targets[i]) != string::npos) {
        if (i == 0 && tmp.size() != 0)
          continue;
        tmp.push_back(extract(targets[i],line));
      }
  }
  found.push_back(tmp);
  tmp.erase(tmp.begin(), tmp.end());
  createCSV(found);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Arguments error!!!\n");
    return 1;
  }
  readFile(getName(argv));
  return 0;
}
