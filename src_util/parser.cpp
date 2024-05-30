#include <iomanip>
#include <cstring>
#include <cstdio>

#include "parser.hpp"
#include "global_parameter.hpp"

using namespace std;

bool parser::no_rewind = false;

/**
 Looks for the string \a search in the input stream and positions the stream right after the string if found.
 If not, returns the stream at EOF.
 Does not start from the beginning of file, but from current position, contrary to the operators defined above.
 */
istream & parser::find_next(istream &flux, const char* search)
{
  string buff;
  
  //recherche la chaine
  while (flux.peek() != EOF)
  {
    flux >> buff;
    if(buff.compare(search)==0){
      return flux;
    }
    flux.ignore(numeric_limits<streamsize>::max(), '\n');
  }
  cerr << "ATTENTION: " << search << " not found in input file.";
  exit(1);
  return flux;
}



/**
 Looks for the string \a search in the input stream and positions the stream right after the string if found.
 If not, returns the stream at EOF.
 Used to input parameters (located after a keyword) or options that are not mandatory
 */
istream & operator==(istream &flux, const char* search)
{
  string buff;
  
  // resets the stream to the beginning
  if(!parser::no_rewind){
    flux.clear();
    flux.seekg(0);
  }
  //recherche la chaine
  while (flux.peek() != EOF)
  {
    flux >> buff;
    if(buff.compare(search)==0){
      return flux;
    }
    flux.ignore(numeric_limits<streamsize>::max(), '\n');
  }
  flux.setstate(ios::failbit);
  return flux;
}



/**
 Looks for the string \a search in the input stream and positions the stream right after the string if found.
 If not, returns the stream at EOF.
 Used to input parameters (located after a keyword) or options that are not mandatory
 */
istream & operator==(istream &flux, const string &search)
{
  string buff;
  
  // resets the stream to the beginning
  if(!parser::no_rewind){
    flux.clear();
    flux.seekg(0);
  }
  //recherche la chaine
  while (flux.peek() != EOF)
  {
    flux >> buff;
    if(buff.compare(search)==0){
      return flux;
    }
    flux.ignore(numeric_limits<streamsize>::max(), '\n');
  }
  flux.setstate(ios::failbit);
  return flux;
}



/**
 Looks for the string \a search in the input stream and positions the stream right after the string if found.
 If not, terminate the program.
 Used to input mandatory parameters (located after a keyword)
 */
istream & operator>>(istream &flux, const char* search)
{
  string buff;
  
  // resets the stream to the beginning
  if(!parser::no_rewind){
    flux.clear();
    flux.seekg(0);
  }
  //recherche la chaine
  while (flux.peek() != EOF)
  {
    flux >> buff;
    if(buff.compare(search)==0){
      return flux;
    }
    flux.ignore(numeric_limits<streamsize>::max(), '\n');
  }
  cerr << "ATTENTION: " << search << " not found in input file.";
  exit(1);
  return flux;
}



/**
 Looks for the string \a search in the input stream and positions the stream right after the string if found.
 If not, terminate the program.
 Used to input mandatory parameters (located after a keyword)
 */
istream & operator>>(istream &flux, const string &search)
{
  string buff;
  
  // resets the stream to the beginning
  if(!parser::no_rewind){
    flux.clear();
    flux.seekg(0);
  }
  //recherche la chaine
  while (flux.peek() != EOF)
  {
    flux >> buff;
    if(buff.compare(search)==0){
      return flux;
    }
    flux.ignore(numeric_limits<streamsize>::max(), '\n');
  }
  cerr << "ATTENTION: " << search << " not found in input file.";
  exit(1);
  return flux;
}



/**
 reads a vector of string from a line. Stops at the first comment character (#)
 clears the vector before reading
 */
vector<string> read_strings(istream &s)
{
  vector<string> X;
  string line;

  do{
    getline(s,line);
    istringstream sin(line);
    while(sin.good()){
      string tmp;
      sin >> skipws >> tmp;
      if(tmp[0]=='#' or tmp.size() == 0) break;
      X.push_back(tmp);
    }
  } while(line.length() > 0 and line[0]=='#');
  return X;
}



ostream & operator<<(ostream &s, vector<string> &X)
{
  for(auto& x: X) s << x << "  ";
  return s;
}



/**
 returns the suffix number of a string. E.g. "abc_12" yields 12.
 ED WARNING : string argument is modified: the suffix is removed.
 */
int cluster_index_from_string(string& S)
{
  int label = 0;
  size_t pos = S.rfind('_');
  if(pos != string::npos){
    label =  from_string<int>(S.substr(pos+1));
    S.erase(pos);
  }
  return label;
}



void check_name(const string& S)
{
  if(S.rfind('_') != string::npos) qcm_ED_throw("the separator '_' is forbidden within operator names!" );
}



/**
 splits a string by delimiter and returns a vector of substrings
 */
vector<std::string> split_string(const string &s, char delim) {
  vector<string> elems;
  std::stringstream ss;
  ss.str(s);
  string item;
  while (getline(ss, item, delim)) elems.push_back(item);
  return elems;
}



/**
 Prints a banner-like message with message \a s, padded with character \a c
 @param c character for padding
 @param s string to print
 @param fout output stream
 */
void banner(const char c, const char s[128], ostream &fout)
{
	size_t i,l,l2,l3;
	
  l = (int)strlen(s);
  if(l==0){
    for(i=0; i<80; ++i) fout << c;
    fout << endl;
    return;
  }
  if(l < 76){
    l2 = 80 - l - 4;
    if(l2%2) l3 = l2/2 + 1;
    else l3 = l2/2;
    l2 = l2/2;
    fout << "\n";
    for(i=0; i<l2; ++i) fout << c;
    fout << "  ";
    fout << s;
    fout << "  ";
    for(i=0; i<l3; ++i) fout << c;
    fout << endl;
  }
  else{
    for(i=0; i<80; ++i) fout << c;
    fout << endl;
    fout << s << endl;
    for(i=0; i<80; ++i) fout << c;
    fout << endl;
  }
}



/**
 Prints a banner-like message with message \a s, padded with character \a c
 @param c character for padding
 @param s string to print
 @param fout output stream
 */
void banner(const char c, const string &s, ostream &fout)
{
	size_t i,l,l2,l3;
	
  l = (int)s.size();
  if(l==0){
    for(i=0; i<80; ++i) fout << c;
    fout << endl;
    return;
  }
  if(l < 76){
    l2 = 80 - l - 4;
    if(l2%2) l3 = l2/2 + 1;
    else l3 = l2/2;
    l2 = l2/2;
    fout << "\n";
    for(i=0; i<l2; ++i) fout << c;
    fout << "  ";
    fout << s;
    fout << "  ";
    for(i=0; i<l3; ++i) fout << c;
    fout << endl;
  }
  else{
    for(i=0; i<80; ++i) fout << c;
    fout << endl;
    fout << s << endl;
    for(i=0; i<80; ++i) fout << c;
    fout << endl;
  }
}



void qcm_throw(const std::string& s)
{
    banner('*', s, std::cerr);
    throw(s);
}



void qcm_ED_throw(const std::string& s)
{
    banner('*', s, std::cerr);
    throw(s);
}


void qcm_warning(const std::string& s)
{
    if(global_bool("verb_warning")) cout << "QCM WARNING : " << s << endl;
}
