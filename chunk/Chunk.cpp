// Chunk.cpp : �������̨Ӧ�ó������ڵ㡣
//


#include "stdafx.h"
#include "stdlib.h"
#include <string.h>
#include <stdio.h>
#include <afxwin.h>
#include "winnls.h"

using namespace std;
#include <map>
#include <set>
#include <vector>
#include <string>
#include <forward_list>
#include <algorithm>

#pragma warning(disable:4996)

//extern bool CreateIdx(char* psUniHZ,char* psInp,char* psIdx,char* psDat,int nType);
bool CreateIdx(char* psUniHZ, char* psInp, char* psIdx, char* psDat, int nType);
void TestNGram();

//for word2vec start
#include <pthread.h>
#pragma comment(lib,"pthreadVC2.lib")

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define posix_memalign(p, a, s) (((*(p)) = _aligned_malloc((s), (a))), *(p) ?0 :errno)

const int vocab_hash_size = 300000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
typedef float real;                    // Precision of float numbers

struct vocab_word {
	long long cn;
	int *point;
	char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING], chunk[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];

char cfreq[MAX_STRING], borderl[MAX_STRING], borderr[MAX_STRING], borderm[MAX_STRING];
char g_psIdx_R[MAX_STRING], g_psDat_R[MAX_STRING], g_psIdx_L[MAX_STRING], g_psDat_L[MAX_STRING], g_psIdx_M[MAX_STRING], g_psDat_M[MAX_STRING];

struct vocab_word *vocab, *ontologyVec;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, chunk_window = 5, min_count = 5, num_threads = 12, min_reduce = 1, ontology = 0;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn2, *syn3, *syn4, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

struct ontologyitem {
	string name;
	int id, r, l;
	vector<int> WordsTreeInHuffman;
public:
	ontologyitem() {
		name = "";
		r = -1;
		l = -1;
	};
};

map<string, set<int>> Ontologydict;
long long nsyn1 = 0;
ontologyitem Ontologytree[400];
int Ontologytreesize = 0, Ontology2vecSize = 0;
long long nOntologyDistanceTarget;
map<long long, vector<long long>> wordid2ontologyvec;
map<string, long long> ontologyname2id;

struct chunk_node {
	long long m_word;
	struct chunk_node * m_next;

	chunk_node() {
		m_next = NULL;
		m_word = 0;
	}
};
char * g_chunk_flag;	//�з����ַ���char/signed char 1 byte - 128~127
chunk_node ** g_chunk;	//chunk hash

int ArgPos(char *str, int argc, char **argv);
void ConvertGBKToUtf8(char* &strGBK);
//for word2vec end

//for word2vec
// Sorts the vocabulary by frequency using word counts
void InitUnigramTable() {
	int a, i;
	double train_words_pow = 0;
	double d1, power = 0.75;
	table = (int *)malloc(table_size * sizeof(int));
	for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
	i = 0;
	d1 = pow(vocab[i].cn, power) / train_words_pow;
	for (a = 0; a < table_size; a++) {
		table[a] = i;
		if (a / (double)table_size > d1) {
			i++;
			d1 += pow(vocab[i].cn, power) / train_words_pow;
		}
		if (i >= vocab_size) i = vocab_size - 1;
	}
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
	int a = 0, ch;
	while (!feof(fin)) {
		ch = fgetc(fin);
		if (ch == 13) continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
			if (a > 0) {
				if (ch == '\n') ungetc(ch, fin);
				break;
			}
			if (ch == '\n') {
				strcpy(word, (char *)"</s>");
				return;
			}
			else continue;
		}
		word[a] = ch;
		a++;
		if (a >= MAX_STRING - 1) a--;   // Truncate too long words
	}
	word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(const char *word) {
	unsigned long long a, hash = 0;
	for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
	hash = hash % vocab_hash_size;
	return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(const char *word) {
	unsigned int hash = GetWordHash(word);
	while (1) {
		if (vocab_hash[hash] == -1) return -1;
		if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
		hash = (hash + 1) % vocab_hash_size;
	}
	return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
	char word[MAX_STRING];
	ReadWord(word, fin);
	if (feof(fin)) return -1;
	return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
	unsigned int hash, length = strlen(word) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
	strcpy(vocab[vocab_size].word, word);
	vocab[vocab_size].cn = 0;
	vocab_size++;
	// Reallocate memory if needed
	if (vocab_size + 2 >= vocab_max_size) {
		vocab_max_size += 1000;
		vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
	}
	hash = GetWordHash(word);
	while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
	vocab_hash[hash] = vocab_size - 1;
	return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
	return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
	int a, size;
	unsigned int hash;
	// Sort the vocabulary and keep </s> at the first position
	qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	size = vocab_size;
	train_words = 0;
	for (a = 0; a < size; a++) {
		// Words occuring less than min_count times will be discarded from the vocab
		if ((vocab[a].cn < min_count) && (a != 0)) {
			vocab_size--;
			free(vocab[a].word);
		}
		else {
			// Hash will be re-computed, as after the sorting it is not actual
			hash = GetWordHash(vocab[a].word);
			while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
			vocab_hash[hash] = a;
			train_words += vocab[a].cn;
		}
	}
	vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
	// Allocate memory for the binary tree construction
	for (a = 0; a < vocab_size; a++) {
		vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
	int a, b = 0;
	unsigned int hash;
	for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
		vocab[b].cn = vocab[a].cn;
		vocab[b].word = vocab[a].word;
		b++;
	}
	else free(vocab[a].word);
	vocab_size = b;
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	for (a = 0; a < vocab_size; a++) {
		// Hash will be re-computed, as it is not actual
		hash = GetWordHash(vocab[a].word);
		while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
		vocab_hash[hash] = a;
	}
	fflush(stdout);
	min_reduce++;
}

int CreateBinarySubTree(vocab_word * vocab, vector<int> intersection, int nsyn1, const vector<vector<int>> path) {
	long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];
	int vocab_size = intersection.size();
	long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
	for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
	pos1 = vocab_size - 1;
	pos2 = vocab_size;

	for (a = 0; a < vocab_size - 1; a++) {
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min1i = pos1;
				pos1--;
			}
			else {
				min1i = pos2;
				pos2++;
			}
		}
		else {
			min1i = pos2;
			pos2++;
		}
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min2i = pos1;
				pos1--;
			}
			else {
				min2i = pos2;
				pos2++;
			}
		}
		else {
			min2i = pos2;
			pos2++;
		}
		count[vocab_size + a] = count[min1i] + count[min2i];
		parent_node[min1i] = vocab_size + a;
		parent_node[min2i] = vocab_size + a;
		binary[min2i] = 1;
	}
	// Now assign binary code to each vocabulary word
	char * strGbk = new char[MAX_STRING];
	for (a = 0; a < vocab_size; a++) {
		b = a;
		i = 0;
		while (1) {
			code[i] = binary[b];
			point[i] = b;
			i++;
			b = parent_node[b];
			if (b == vocab_size * 2 - 2) break;
		}

		int codelen = i + path.size();
		vocab[intersection[a]].codelen = codelen;
		for (b = 0; b < path.size(); b++) {
			vocab[intersection[a]].code[b] = path[b][1];
			vocab[intersection[a]].point[b] = path[b][0];
		}
		vocab[intersection[a]].point[path.size()] = vocab_size - 2 + nsyn1;
		for (b = 0; b < i; b++) {
			vocab[intersection[a]].code[i - b - 1 + path.size()] = code[b];
			vocab[intersection[a]].point[i - b + path.size()] = point[b] - vocab_size + nsyn1;
		}
	}
	free(count);
	free(binary);
	free(parent_node);
	delete strGbk;
	return vocab_size * 2 - 2;
}

void LearnVocabFromTrainFile() {
	char word[MAX_STRING];
	FILE *fin;
	long long a, i;
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	vocab_size = 0;
	AddWordToVocab((char *)"</s>");
	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;
		train_words++;
		if ((debug_mode > 1) && (train_words % 100000 == 0)) {
			printf("%lldK%c", train_words / 1000, 13);
			fflush(stdout);
		}
		i = SearchVocab(word);
		if (i == -1) {
			a = AddWordToVocab(word);
			vocab[a].cn = 1;
		}
		else vocab[i].cn++;
		if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
	}
	SortVocab();
	if (debug_mode > 0) {
		printf("Total Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	file_size = ftell(fin);
	fclose(fin);
}

void SaveVocab() {
	long long i;
	FILE *fo = fopen(save_vocab_file, "wb");
	for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
	fclose(fo);
}

void ReadVocab() {
	long long a, i = 0;
	char c;
	char word[MAX_STRING];
	FILE *fin = fopen(read_vocab_file, "rb");
	if (fin == NULL) {
		printf("Vocabulary file not found\n");
		exit(1);
	}
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	vocab_size = 0;
	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;
		a = AddWordToVocab(word);
		fscanf(fin, "%lld%c", &vocab[a].cn, &c);
		i++;
	}
	SortVocab();
	if (debug_mode > 0) {
		printf("Total Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	fseek(fin, 0, SEEK_END);
	file_size = ftell(fin);
	fclose(fin);
}

void ReadChunk(char * file) {
	
	FILE* pf;
	long long curWordID = 0, i = 0, t = 0;
	char * strUtf8 = new char[MAX_STRING];
	char szBuff[MAX_STRING];
	char szWord[MAX_STRING];
	int curbinary = 0; chunk_node * cnt = NULL;
	bool flag = false;
	g_chunk_flag = new char[ceil(vocab_size / 8) + 1];
	g_chunk = new chunk_node*[vocab_size];
	for (size_t i = 0; i < vocab_size; i++) g_chunk[i] = NULL;
	for (size_t i = 0; i < ceil(vocab_size / 8) + 1; i++) g_chunk_flag[i] = 0;

	pf = fopen(file, "rb");
	while (fgets(szBuff, MAX_STRING, pf) != NULL) {
		if (szBuff[strlen(szBuff) - 1] == 0x0a) {
			szBuff[strlen(szBuff) - 1] = 0;
		}
		if (szBuff[strlen(szBuff) - 1] == 0x0d) {
			szBuff[strlen(szBuff) - 1] = 0;
		}
		if (!strlen(szBuff)) continue;

		if (szBuff[0] == '(' || szBuff[0] == '[') { flag = true; continue; }
		else if (szBuff[0] == ')' || szBuff[0] == ']') { flag = false; continue; }
		if (szBuff[0] == '/' || szBuff[1] == '/') continue;

		if (!flag) {
			//read cat_word
			strcpy(strUtf8, strchr(szBuff, '_') + 1);
			ConvertGBKToUtf8(strUtf8);
			curWordID = SearchVocab(strUtf8);
			if (curWordID == -1) continue;
			curbinary = (int)pow(2, (double)(curWordID % 8));
			g_chunk_flag[curWordID / 8] = g_chunk_flag[curWordID / 8] | curbinary;
		}
		else {
			//read list
			t = strchr(szBuff, ' ') - szBuff;
			strncpy(strUtf8, szBuff, t);
			strUtf8[t] = 0;
			ConvertGBKToUtf8(strUtf8);
			i = SearchVocab(strUtf8);
			if (i == -1) continue;
			if (curWordID == -1) continue;
			//build  g_chunk			
			cnt = g_chunk[curWordID];
			g_chunk[curWordID] = new chunk_node();
			g_chunk[curWordID]->m_next = cnt;
			g_chunk[curWordID]->m_word = i;
			cnt = NULL;
		}
	}

	fclose(pf);
	printf("end readChunk\n");
}

void DeleteChunk() {
	delete g_chunk_flag;

	chunk_node * cnt;
	for (size_t i = 0; i < vocab_size; i++)
	{
		while (g_chunk[i])
		{
			cnt = g_chunk[i];
			g_chunk[i] = g_chunk[i]->m_next;
			delete cnt;
		}
	}
	delete g_chunk;
}

const void* buildHuffman(int tree, vector<vector<int>> path, string param = "")
{
	//if(debug_mode > 1) printf("\n=-=-\n%s\t\t%d\t%d\n", Ontologytree[tree].name.c_str(), Ontologytree[tree].l, Ontologytree[tree].r);
	if (Ontologytree[tree].name.find("isIn:") == std::string::npos) return 0;

	string ontoname = Ontologytree[tree].name.substr(5);
	set<int>::iterator it;
	vector<int> t;
	for (it = Ontologydict[ontoname.c_str()].begin();
		it != Ontologydict[ontoname.c_str()].end();
		++it) 
		t.push_back(*it);

	//if (debug_mode > 1) for (int i = 0; i < path.size(); i++) printf("{%d,%d},\n", path[i][0],path[i][1]);
	path.push_back(vector<int>{tree, 1});
	nsyn1 += CreateBinarySubTree(vocab, t, nsyn1, path);
	return 0;
}

const void* buildOntologyVec(int tree, vector<vector<int>> path, string param = "")
{
	if (Ontologytree[tree].name.find("isIn:") == std::string::npos) return 0;

	string ontoname = Ontologytree[tree].name.substr(5);
	set<int>::iterator it;

	//if (debug_mode > 1) for (int i = 0; i < path.size(); i++) printf("{%d,%d},\n", path[i][0],path[i][1]);
	path.push_back(vector<int>{tree, 1});
	ontologyVec[Ontology2vecSize].word = (char *)calloc(ontoname.length(), sizeof(char));		//ע���������
	ontologyVec[Ontology2vecSize].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
	ontologyVec[Ontology2vecSize].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	ontologyVec[Ontology2vecSize].cn = Ontology2vecSize;
	ontologyname2id[ontoname] = Ontology2vecSize;

	strcpy(ontologyVec[Ontology2vecSize].word, ontoname.c_str());
	ontologyVec[Ontology2vecSize].codelen = path.size();

	for (int b = 0; b < path.size(); b++) {
		ontologyVec[Ontology2vecSize].code[b] = path[b][1];
		ontologyVec[Ontology2vecSize].point[b] = path[b][0];
	}
	Ontology2vecSize++;
	return 0;
}

void PreOrderTraversalOntologyTree(int root, ontologyitem * tree, vector<vector<int>> path, const void *(*op)(int tree, vector<vector<int>> path, string param)) {
	if (root == -1) return;
	vector<vector<int>> curpath;
	op(root, path, "");
	curpath = path; curpath.push_back(vector<int>{root, 0});
	PreOrderTraversalOntologyTree(tree[root].l, tree, curpath, op);
	curpath = path; curpath.push_back(vector<int>{root, 1});
	PreOrderTraversalOntologyTree(tree[root].r, tree, curpath, op);
}

bool createOntologytree() {
	FILE* pf;
	pf = fopen("HowNet_Taxonomy_Entity.txt", "rb");
	if (pf == NULL)
		return false;
	char szBuff[MAX_SENTENCE_LENGTH];
	char szWord[MAX_SENTENCE_LENGTH];

	szWord[0] = 0;
	szBuff[0] = 0;
	string currentline;
	while (fgets(szBuff, MAX_SENTENCE_LENGTH, pf) != NULL) {
		if (szBuff[strlen(szBuff) - 1] == 0x0a) {
			szBuff[strlen(szBuff) - 1] = 0;
		}
		if (szBuff[strlen(szBuff) - 1] == 0x0d) {
			szBuff[strlen(szBuff) - 1] = 0;
		}
		currentline = szBuff;
		std::size_t foundl = currentline.find("{");
		std::size_t foundr = currentline.find("}");
		if (foundl == std::string::npos) continue;
		Ontologytree[Ontologytreesize].id = foundl;
		Ontologytree[Ontologytreesize].name = currentline.substr(foundl + 1, foundr - foundl - 1);
		Ontologytreesize++;

		Ontologytree[Ontologytreesize].id = foundl + 1;
		Ontologytree[Ontologytreesize].name = "isIn:" + currentline.substr(foundl + 1, foundr - foundl - 1);
		Ontologytreesize++;
	}
	Ontologytree[Ontologytreesize].id = -1;
	Ontologytree[Ontologytreesize].name = "END";
	Ontologytreesize++;
	fclose(pf);
	for (int i = 0; i < Ontologytreesize; i++) {
		for (size_t j = i + 1; j < Ontologytreesize; j++)
		{
			if (Ontologytree[j].id > Ontologytree[i].id) {
				Ontologytree[i].l = j;
				break;
			}
			else {
				Ontologytree[i].l = -1;
				break;
			}
		}
		for (size_t j = i + 1; j < Ontologytreesize; j++)
		{
			if (Ontologytree[j].id == Ontologytree[i].id) {
				Ontologytree[i].r = j;
				break;
			}
			else if (Ontologytree[j].id < Ontologytree[i].id) {
				Ontologytree[i].r = -1;
				break;
			}
		}
	}

	map<string, string> distinctword2dictname;
	set<string> qiyici;
	string currentword;
	pf = fopen("EntityVocabs.txt", "rb");
	char * strUtf8 = new char[MAX_SENTENCE_LENGTH];
	vector<string> t;
	int i = 0, j = 0;
	while (fgets(szBuff, MAX_SENTENCE_LENGTH, pf) != NULL) {
		if (szBuff[strlen(szBuff) - 1] == 0x0a) szBuff[strlen(szBuff) - 1] = 0;
		if (szBuff[strlen(szBuff) - 1] == 0x0d) szBuff[strlen(szBuff) - 1] = 0;
		if (!strlen(szBuff)) continue;
		if (szBuff[0] == 'D' && szBuff[1] == 'E' && szBuff[2] == 'F') continue;
		if (szBuff[0] == 0) continue;
		if (szBuff[0] == '#') {
			currentword = &szBuff[1];
			continue;
		}
		strcpy(strUtf8, szBuff);
		ConvertGBKToUtf8(strUtf8);

		i = SearchVocab(strUtf8);
		if (i == -1) continue;
		if (i == j) continue; j = i;

		map<string, string>::iterator it;

		it = distinctword2dictname.find(szBuff);
		if (it != distinctword2dictname.end()) {
			Ontologydict[it->second].erase(i);
		}
		else {
			Ontologydict[currentword].insert(i);
			distinctword2dictname[szBuff] = currentword;
		}
	}
	nsyn1 = Ontologytreesize;
	//root vector<vector<int>>: [[14entity,0],[24thing, 0], [34physical, 0]]
	PreOrderTraversalOntologyTree(0, Ontologytree, vector<vector<int>>{}, buildHuffman);
	fclose(pf);
	if (debug_mode > 0) {
		printf("OntologyTree size: %lld\n", Ontologytreesize);
		printf("Words in OntologyTree: %lld\n", distinctword2dictname.size());
	}
	fflush(stdout);
	delete strUtf8;
	return true;
}

bool createOntology2Vec() {
	int nleaves = 0;
	FILE* pf;
	pf = fopen("HowNet_Taxonomy_Entity.txt", "rb");
	if (pf == NULL)
		return false;
	char szBuff[MAX_STRING];
	char szWord[MAX_STRING];

	szWord[0] = 0;
	szBuff[0] = 0;
	string currentline;
	while (fgets(szBuff, MAX_STRING, pf) != NULL) {
		if (szBuff[strlen(szBuff) - 1] == 0x0a) {
			szBuff[strlen(szBuff) - 1] = 0;
		}
		if (szBuff[strlen(szBuff) - 1] == 0x0d) {
			szBuff[strlen(szBuff) - 1] = 0;
		}
		currentline = szBuff;
		std::size_t foundl = currentline.find("{");
		std::size_t foundr = currentline.find("}");
		if (foundl == std::string::npos) continue;
		Ontologytree[Ontologytreesize].id = foundl;
		Ontologytree[Ontologytreesize].name = currentline.substr(foundl + 1, foundr - foundl - 1);
		Ontologytreesize++;

		Ontologytree[Ontologytreesize].id = foundl + 1;
		Ontologytree[Ontologytreesize].name = "isIn:" + currentline.substr(foundl + 1, foundr - foundl - 1);
		Ontologytreesize++;
	}
	Ontologytree[Ontologytreesize].id = -1;
	Ontologytree[Ontologytreesize].name = "END";
	Ontologytreesize++;
	fclose(pf);
	for (int i = 0; i < Ontologytreesize; i++) {
		for (size_t j = i + 1; j < Ontologytreesize; j++)
		{
			if (Ontologytree[j].id > Ontologytree[i].id) {
				Ontologytree[i].l = j;
				break;
			}
			else {
				Ontologytree[i].l = -1;
				break;
			}
		}
		for (size_t j = i + 1; j < Ontologytreesize; j++)
		{
			if (Ontologytree[j].id == Ontologytree[i].id) {
				Ontologytree[i].r = j;
				break;
			}
			else if (Ontologytree[j].id < Ontologytree[i].id) {
				Ontologytree[i].r = -1;
				break;
			}
		}
	}

	nsyn1 = Ontologytreesize;
	Ontology2vecSize = 0;
	//root vector<vector<int>>: [[14entity,0],[24thing, 0], [34physical, 0]]
	PreOrderTraversalOntologyTree(0, Ontologytree, vector<vector<int>>{}, buildOntologyVec);

	string currentword;
	pf = fopen("EntityVocabs.txt", "rb");
	char * strUtf8 = new char[MAX_STRING];
	vector<string> t;
	int i = 0;
	while (fgets(szBuff, MAX_STRING, pf) != NULL) {
		if (szBuff[strlen(szBuff) - 1] == 0x0a) {
			szBuff[strlen(szBuff) - 1] = 0;
		}
		if (szBuff[strlen(szBuff) - 1] == 0x0d) {
			szBuff[strlen(szBuff) - 1] = 0;
		}
		if (!strlen(szBuff)) continue;
		if (szBuff[0] == 'D' && szBuff[1] == 'E' && szBuff[2] == 'F') {
			continue;
		}
		if (szBuff[0] == '#') {
			currentword = &szBuff[1];
			continue;
		}
		strcpy(strUtf8, szBuff);
		ConvertGBKToUtf8(strUtf8);

		i = SearchVocab(strUtf8);
		if (i == -1) continue;
		wordid2ontologyvec[i].push_back(ontologyname2id[currentword]);
	}
	fclose(pf);
	if (debug_mode > 0) {
		printf("OntologyTree size: %lld\n", Ontologytreesize);
		printf("wordid2ontologyvec size: %lld\n", wordid2ontologyvec.size());
	}
	fflush(stdout);
	return true;
}

void InitNet() {
	long long a, b, c, d;
	unsigned long long next_random = 1;
	a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
	c = posix_memalign((void **)&syn2, 128, (long long)Ontology2vecSize * layer1_size * sizeof(real));
	if (strlen(chunk)) posix_memalign((void **)&syn3, 128, (long long)vocab_size * layer1_size * sizeof(real));
	if (strlen(chunk)) posix_memalign((void **)&syn4, 128, (long long)(vocab_size * 2 + Ontologytreesize) * layer1_size * sizeof(real));
	if (syn0 == NULL) { printf("syn0 Memory allocation failed\n"); exit(1); }
	if (hs) {
		a = posix_memalign((void **)&syn1, 128, (long long)(vocab_size * 2 + Ontologytreesize) * layer1_size * sizeof(real));
		if (syn1 == NULL) { printf("syn1 Memory allocation failed\n"); exit(1); }
		for (a = 0; a < (vocab_size * 2 + Ontologytreesize); a++) for (b = 0; b < layer1_size; b++)
			syn1[a * layer1_size + b] = 0;
	}
	if (negative>0) {
		a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (syn1neg == NULL) { printf("syn1neg Memory allocation failed\n"); exit(1); }
		for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
			syn1neg[a * layer1_size + b] = 0;
	}
	if (strlen(chunk)) for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
		next_random = next_random * (unsigned long long)25214903917 + 11;
		syn3[a * layer1_size + b] = syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
	}
	if (strlen(chunk)) for (a = 0; a < (vocab_size * 2); a++) for (b = 0; b < layer1_size; b++) {
		syn4[a * layer1_size + b] = 0;
	}
	for (a = 0; a < Ontology2vecSize; a++) for (b = 0; b < layer1_size; b++) {
		next_random = next_random * (unsigned long long)25214903917 + 11;
		syn2[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
	}

	vector<int> first, second, v(vocab_size);
	vector<int>::iterator it;

	for (int i = 0; i < vocab_size; i++) first.push_back(i);
	for (map<string, set<int>>::iterator it1 = Ontologydict.begin(); it1 != Ontologydict.end(); ++it1)
		for (set<int>::iterator it2 = it1->second.begin(); it2 != it1->second.end(); ++it2)
			second.push_back(*it2);

	sort(first.begin(), first.end());
	sort(second.begin(), second.end());

	it = set_difference(first.begin(), first.end(), second.begin(), second.end(), v.begin());

	v.resize(it - v.begin());

	vector<vector<int>> root;//[[14entity,0],[24thing, 0], [34physical, 0]]
	root.push_back(vector<int>{0, 1});

	nsyn1 += CreateBinarySubTree(vocab, v, nsyn1, root);
}

bool isInChunk(long long word, long long chunk_word) {
	int curbinary = (int)pow(2, (double)(word % 8));
	if (g_chunk_flag[word / 8] & curbinary) {
		chunk_node * cnt = g_chunk[word];
		while (cnt){
			if (cnt->m_word == chunk_word) return true;
			cnt = cnt->m_next;
		}
	}
	return false;
}

bool isTransBoundary(long long * sen, int word,  int chunk) {
	char psSent[MAX_SENTENCE_LENGTH], psSentGbk[MAX_SENTENCE_LENGTH];
	int n = 0;
	for (size_t i = min(word, chunk); i < max(word, chunk); i++) n += sprintf(psSent + n, "%s", vocab[i].word);
	UTF8_2_GBK(psSent, psSentGbk);
	int nLen = strlen(psSent) / 2;
	for (int nPos = 1; nPos<nLen; nPos++) {
		float fProb = g_HZNGram.GetSentenceBreakProb(psSentGbk, nPos, 6, 0.0);
		printf("%d\t%s\t%f\n\n", nPos, psSentGbk, fProb);
	}
	return true;
}	

void *TrainModelThread(void *id) {
	long long a, b, d, i, j, cw, word, chunk_word, last_word, sentence_length = 0, sentence_position = 0;
	long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1], curonto = 0;
	long long l1, l2, c, target, label, local_iter = iter;
	unsigned long long next_random = (long long)id;
	real f, g;
	clock_t now;
	real *neu1 = (real *)calloc(layer1_size, sizeof(real));
	real *neu1e = (real *)calloc(layer1_size, sizeof(real));
	FILE *fi = fopen(train_file, "rb");
	fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
	while (1) {
		if (word_count - last_word_count > 10000) {
			word_count_actual += word_count - last_word_count;
			last_word_count = word_count;
			if ((debug_mode > 1)) {
				now = clock();
				printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
					word_count_actual / (real)(iter * train_words + 1) * 100,
					word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
				fflush(stdout);
			}
			alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
			if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
		}
		if (sentence_length == 0) {
			while (1) {
				word = ReadWordIndex(fi);
				if (feof(fi)) break;
				if (word == -1) continue;
				word_count++;
				if (word == 0) break;
				// The subsampling randomly discards frequent words while keeping the ranking same
				if (sample > 0) {
					real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
					next_random = next_random * (unsigned long long)25214903917 + 11;
					if (ran < (next_random & 0xFFFF) / (real)65536) continue;
				}
				sen[sentence_length] = word;
				sentence_length++;
				if (sentence_length >= MAX_SENTENCE_LENGTH) break;
			}
			sentence_position = 0;
		}
		if (feof(fi) || (word_count > train_words / num_threads)) {
			word_count_actual += word_count - last_word_count;
			local_iter--;
			if (local_iter == 0) break;
			word_count = 0;
			last_word_count = 0;
			sentence_length = 0;
			fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
			continue;
		}
		word = sen[sentence_position];
		if (word == -1) continue;
		for (c = 0; c < layer1_size; c++) neu1[c] = 0;
		for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
		next_random = next_random * (unsigned long long)25214903917 + 11;
		b = next_random % window;
		if (cbow) {  //train the cbow architecture
					 // in -> hidden
			cw = 0;
			for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
				c = sentence_position - window + a;
				if (c < 0) continue;
				if (c >= sentence_length) continue;
				last_word = sen[c];
				if (last_word == -1) continue;
				for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
				cw++;
			}
			if (cw) {
				for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
				if (hs) for (d = 0; d < vocab[word].codelen; d++) {
					f = 0;
					l2 = vocab[word].point[d] * layer1_size;
					// Propagate hidden -> output
					for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
					if (f <= -MAX_EXP) continue;
					else if (f >= MAX_EXP) continue;
					else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
					// 'g' is the gradient multiplied by the learning rate
					g = (1 - vocab[word].code[d] - f) * alpha;
					// Propagate errors output -> hidden
					for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
					// Learn weights hidden -> output
					for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
				}
				// NEGATIVE SAMPLING
				if (negative > 0) for (d = 0; d < negative + 1; d++) {
					if (d == 0) {
						target = word;
						label = 1;
					}
					else {
						next_random = next_random * (unsigned long long)25214903917 + 11;
						target = table[(next_random >> 16) % table_size];
						if (target == 0) target = next_random % (vocab_size - 1) + 1;
						if (target == word) continue;
						label = 0;
					}
					l2 = target * layer1_size;
					f = 0;
					for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
					if (f > MAX_EXP) g = (label - 1) * alpha;
					else if (f < -MAX_EXP) g = (label - 0) * alpha;
					else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
					for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
					for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
				}
				// hidden -> in
				for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
					c = sentence_position - window + a;
					if (c < 0) continue;
					if (c >= sentence_length) continue;
					last_word = sen[c];
					if (last_word == -1) continue;
					for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
				}
			}
		}
		else {  //train skip-gram
			for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
				c = sentence_position - window + a;
				if (c < 0) continue;
				if (c >= sentence_length) continue;
				last_word = sen[c];
				if (last_word == -1) continue;
				l1 = last_word * layer1_size;
				for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
				// HIERARCHICAL SOFTMAX
				if (hs) for (d = 0; d < vocab[word].codelen; d++) {
					f = 0;
					l2 = vocab[word].point[d] * layer1_size;
					// Propagate hidden -> output
					for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
					if (f <= -MAX_EXP) continue;
					else if (f >= MAX_EXP) continue;
					else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
					// 'g' is the gradient multiplied by the learning rate
					g = (1 - vocab[word].code[d] - f) * alpha;
					// Propagate errors output -> hidden
					for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
					// Learn weights hidden -> output
					for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
				}
				// NEGATIVE SAMPLING
				if (negative > 0) for (d = 0; d < negative + 1; d++) {
					if (d == 0) {
						target = word;
						label = 1;
					}
					else {
						next_random = next_random * (unsigned long long)25214903917 + 11;
						target = table[(next_random >> 16) % table_size];
						if (target == 0) target = next_random % (vocab_size - 1) + 1;
						if (target == word) continue;
						label = 0;
					}
					l2 = target * layer1_size;
					f = 0;
					for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
					if (f > MAX_EXP) g = (label - 1) * alpha;
					else if (f < -MAX_EXP) g = (label - 0) * alpha;
					else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
					for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
					for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
				}
				// Learn weights input -> hidden
				if ((hs)&& (negative > 0))  for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];

				if (ontology == 2) {
					for (i = 0; i < wordid2ontologyvec[word].size(); i++) {
						curonto = wordid2ontologyvec[word][i];
						l1 = curonto * layer1_size;
						for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
						for (d = 0; d < ontologyVec[curonto].codelen; d++) {
							f = 0;
							l2 = ontologyVec[curonto].point[d] * layer1_size;
							// Propagate hidden -> output
							for (c = 0; c < layer1_size; c++) f += syn2[c + l1] * syn1[c + l2];
							if (f <= -MAX_EXP) continue;
							else if (f >= MAX_EXP) continue;
							else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
							// 'g' is the gradient multiplied by the learning rate
							g = (1 - ontologyVec[curonto].code[d] - f) * alpha;
							// Propagate errors output -> hidden
							for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
							// Learn weights hidden -> output
							for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn2[c + l1];
						}
						for (c = 0; c < layer1_size; c++) syn2[c + l1] += neu1e[c];
					}
				}

				//-chunk VPBin.txt
				if (chunk[0]) {
					for (i = 0; i < chunk_window * 2 + 1; i++) {
						if (i == chunk_window) continue;
						for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
						c = sentence_position - chunk_window + i;
						if (c < 0) continue;
						if (c >= sentence_length) continue;
						if (c == (sentence_position - window + a)) continue;

						chunk_word = sen[c];
						if (!isInChunk(word, chunk_word)) continue;
						//if (!isTransBoundary(sen, sentence_position, c)) continue;

						for (d = 0; d < vocab[chunk_word].codelen; d++) {
							f = 0;
							l2 = vocab[chunk_word].point[d] * layer1_size;
							for (c = 0; c < layer1_size; c++) f += syn3[c + l1] * syn4[c + l2];
							if (f <= -MAX_EXP) continue;
							else if (f >= MAX_EXP) continue;
							else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
							g = (1 - vocab[chunk_word].code[d] - f) * alpha;
							for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn4[c + l2];
							for (c = 0; c < layer1_size; c++) syn4[c + l2] += g * syn3[c + l1];
						}
						for (c = 0; c < layer1_size; c++) syn3[c + l1] += neu1e[c];
					}
				}
			}
		}
		sentence_position++;
		if (sentence_position >= sentence_length) {
			sentence_length = 0;
			continue;
		}
	}
	fclose(fi);
	free(neu1);
	free(neu1e);
	pthread_exit(NULL);
	return id;
}

void TrainModel() {
	long a, b, c, d;
	FILE *fo;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	printf("Starting training using file %s\n", train_file);
	starting_alpha = alpha;
	if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
	if (save_vocab_file[0] != 0) SaveVocab();
	if (output_file[0] == 0) return;
	if (strlen(chunk)) ReadChunk(chunk);

	//ontology = 0	��ʹ�ñ�������default
	//ontology = 1	������+���ڵ�huffman
	//ontology = 2	��������
	switch (ontology)
	{
	case 1:
		if (!createOntologytree()) return;
		else break;
	case 2:
		if (!createOntology2Vec()) return;
		else break;
	default:
		break;
	}

	InitNet();
	if (negative > 0) InitUnigramTable();
	start = clock();
	for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

	fo = fopen(output_file, "wb");
	fprintf(fo, "%lld %lld %lld %lld", vocab_size, layer1_size, chunk_window, Ontologytreesize);
	for (a = 0; a < vocab_size; a++) {
		if (a == 79)
			int d = 0;
		fprintf(fo, "%s ", vocab[a].word);
		fwrite(&vocab[a].cn, sizeof(long long), 1, fo);
		fwrite(&vocab[a].codelen, sizeof(char), 1, fo);
		for (b = 0; b < vocab[a].codelen; b++) fwrite(&vocab[a].point[b], sizeof(int), 1, fo);
		for (b = 0; b < vocab[a].codelen; b++) fwrite(&vocab[a].code[b], sizeof(char), 1, fo);
		for (b = 0; b < layer1_size; b++) fwrite(&syn3[a * layer1_size + b], sizeof(real), 1, fo);
	}
	for (a = 0; a < vocab_size * 2 + Ontologytreesize; a++) {
		for (b = 0; b < layer1_size; b++) fwrite(&syn4[a * layer1_size + b], sizeof(real), 1, fo);
	}
	fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

void ConvertGBKToUtf8(char* &strGBK)
{
	int len = MultiByteToWideChar(CP_ACP, 0, (LPCCH)strGBK, -1, NULL, 0);
	wchar_t * wszUtf8 = new wchar_t[len];
	memset(wszUtf8, 0, len);
	MultiByteToWideChar(CP_ACP, 0, (LPCCH)strGBK, -1, wszUtf8, len);
	len = WideCharToMultiByte(CP_UTF8, 0, wszUtf8, -1, NULL, 0, NULL, NULL);
	char *szUtf8 = new char[len + 1];
	memset(szUtf8, 0, len + 1);
	WideCharToMultiByte(CP_UTF8, 0, wszUtf8, -1, szUtf8, len, NULL, NULL);
	strcpy(strGBK, szUtf8);
	delete[] szUtf8;
	delete[] wszUtf8;
}

int _tmain(int argc, _TCHAR* argv[])
{
	output_file[0] = 0;
	chunk[0] = 0;
	save_vocab_file[0] = 0;
	read_vocab_file[0] = 0;
	int i = 0;
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
	if (cbow) alpha = 0.05;
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-chunk-window", argc, argv)) > 0) chunk_window = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-ontology", argc, argv)) > 0) ontology = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-chunk", argc, argv)) > 0) strcpy(chunk, argv[i + 1]);
	if ((i = ArgPos((char *)"-cfreq", argc, argv)) > 0) strcpy(cfreq, argv[i + 1]);
	if ((i = ArgPos((char *)"-borderl", argc, argv)) > 0) strcpy(borderl, argv[i + 1]);
	if ((i = ArgPos((char *)"-borderr", argc, argv)) > 0) strcpy(borderr, argv[i + 1]);
	if ((i = ArgPos((char *)"-borderm", argc, argv)) > 0) strcpy(borderm, argv[i + 1]);
	vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
	ontologyVec = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	for (i = 0; i < EXP_TABLE_SIZE; i++) {
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}

	TrainModel();
	return 0;
}