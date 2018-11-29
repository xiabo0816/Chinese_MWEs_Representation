// Chunk_distance.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "../cfl/web.h"

#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <map>
#include <algorithm>

using namespace std;

#define MAXLEN 10240
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_STRING 100
#define MAX_CODE_LENGTH 40

#define bestN 100

#define posix_memalign(p, a, s) (((*(p)) = _aligned_malloc((s), (a))), *(p) ?0 :errno)

//-----------typedef-----------
typedef float real;                    // Precision of float numbers
struct vocab_word {
	long long cn;
	int *point;
	char *word, *code, codelen;
};

int ArgPos(char *str, int argc, char **argv);
void ReadWord(char *word, FILE *fin);
int GetWordHash(const char *word);
int SearchVocab(const char *word);
int AddWordToVocab(char *word, long long vocab_pos);
string JoinString(const char* psSeparator, const vector<string> vItem);
void SplitString(const char* psStr, const char* psSeparator, vector<string> & vItem);
void ConvertUtf8ToGBK(char* &strUtf8);
void ConvertGBKToUtf8(char* &strGBK);
//-----------typedef-----------

//-----------GLOBALS-----------
real *syn_chunk, *syn_internode, *expTable;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
char chunk_file[MAX_STRING], port[MAX_STRING];
struct vocab_word *vocab;
const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
int *vocab_hash, debug_mode = 2, chunk_window = 1, iter = 5, ontology_size = 0;

//-----------GLOBALS-----------
class CQuery
{
public:
	CQuery() {
		m_direction = true;
		m_query[0] = 0;
		m_head = 0;
		m_type = 0;
		//type=0 singleDirection
		//type=1 biDirection
	}

	~CQuery() {
	}
public:
	bool m_direction;
	char m_query[MAXLEN];
	int m_head;
	int m_type;
	map<string, float> m_result;
};
class CQueryNeighbors
{
public:
	CQueryNeighbors() {
		m_head[0] = 0;
		m_dep[0] = 0;
	}

	~CQueryNeighbors() {
	}
public:
	char m_head[MAXLEN];
	char m_dep[MAXLEN];
	map<string, float> m_result;
};
class CQuerySemantic
{
public:
	CQuerySemantic() {
		m_first[0] = 0;
		m_second[0] = 0;
		m_input[0] = 0;
	}

	~CQuerySemantic() {
	}
public:
	char m_first[MAXLEN];
	char m_second[MAXLEN];
	char m_input[MAXLEN];
	map<string, float> m_result;
};

void P_chunk(CQuery *src) {
	int position = -1;
	int *cl = (int *)calloc(vocab_size, sizeof(int));
	//Distance(target, res);
	vector<string> vItem;
	vector<long long> sen;
	SplitString((*src).m_query, " ", vItem);

	long long a, b, d, i, j, cw, word, last_word, chunk_word, sentence_length = vItem.size(), sentence_position = 0;
	long long word_count = 0, last_word_count = 0;
	long long l1 = 0, l2, c, label, local_iter = iter;
	real f, g = 1, gmax = -999, gmin = 999;

	char * strUtf8 = new char[MAXLEN];

	for (size_t i = 0; i < sentence_length; i++)
	{
		strcpy(strUtf8, vItem[i].c_str());
		ConvertGBKToUtf8(strUtf8);
		for (b = 0; b < vocab_size; b++) if (!strcmp(vocab[b].word, strUtf8)) break;

		if (b == vocab_size) b = -1;
		if (b == -1) {
			printf("%s Out of dictionary word!\n", vItem[i].c_str());
			return;
		}
		sen.push_back(b);
	}

	printf("\n\t");
	float *m = new float[sentence_length * sentence_length];
	for (a = 0; a < sentence_length * sentence_length; a++) m[a] = 0;

	for (a = 0; a < sentence_length; a++) printf("%s\t", vItem[a].c_str());
	for (a = 0; a < sentence_length; a++) {
		printf("\n\n%s\t", vItem[a].c_str());
		for (long long k = 0; k < (a - chunk_window); k++) printf("\t");

		for (i = 0; i < chunk_window * 2 + 1; i++) {
			if (i == chunk_window) {
				printf("\t");
				continue;
			}
			c = a - chunk_window + i;
			if (c < 0) continue;
			if (c >= sentence_length) continue;
			chunk_word = sen[c];	//chunk_word 是 a 的搭配
									//两个词互为搭配
									//a是b的搭配
									//b是a的搭配
									//printf("%s=>%s", vItem[a].c_str(), vItem[c].c_str());
			g = 0;
			l1 = sen[a] * layer1_size;
			/*
			for (d = 0; d < vocab[chunk_word].codelen; d++) {
				l2 = vocab[chunk_word].point[d] * layer1_size;
				f = 0;
				for (c = 0; c < layer1_size; c++) f += syn_chunk[c + l1] * syn_internode[c + l2];
				if (f <= -MAX_EXP) continue;
				else if (f >= MAX_EXP) continue;
				else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
				if (vocab[chunk_word].code[d]) g += log(1 - f);
				else g += log(f);
			}
			g = -100*(1 / g);
			if (g > gmax) gmax = g;
			if (g < gmin) gmin = g;
			m[a*sentence_length + a - chunk_window + i] = g;
			*/
				
			for (b = 0; b < sentence_length; b++) {
				l1 = sen[b] * layer1_size;
				if (b == a || b == c) continue;
				for (d = 0; d < vocab[chunk_word].codelen; d++) {
					l2 = vocab[chunk_word].point[d] * layer1_size;
					f = 0;
					for (c = 0; c < layer1_size; c++) f += syn_chunk[c + l1] * syn_internode[c + l2];
					if (f <= -MAX_EXP) continue;
					else if (f >= MAX_EXP) continue;
					else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
					if (vocab[chunk_word].code[d]) g += log(1 - f);
					else g += log(f);
				}
			}
			m[a*sentence_length + a - chunk_window + i] = fabs(g / sentence_length);
			
			printf("%2.3f\t", fabs(g / sentence_length));
		}
	}
	printf("\ngmax\n%2.3f\n", gmax);
	printf("\ngmin\n%2.3f\n", gmin);
	if ((*src).m_type == 0) {
		for (size_t i = 0; i < sentence_length; i++)
		{
			for (size_t j = i + 1; j < sentence_length; j++)
			{
				if (m[i*sentence_length + j]) {
					printf("\n%s-%s:\t%f", vItem[i].c_str(), vItem[j].c_str(), fabs(m[i*sentence_length + j] - m[j*sentence_length + i]));
					//(*src).push_back(vItem[i]);
					//(*src).push_back(vItem[j]);
					//sprintf(strUtf8, "%f", fabs(m[i*sentence_length + j] - m[j*sentence_length + i]));
					//(*src).m_result.insert(pair<string, float>(vItem[i], m[i*sentence_length + j]));
				}
			}
		}
	}

	if ((*src).m_type == 1) {
		i = (*src).m_head;
		for (size_t j = 0; j < sentence_length; j++)
		{
			if (m[i*sentence_length + j] && i != j) {
				(*src).m_result.insert(pair<string, float>(vItem[j], m[i*sentence_length + j]));
			}
		}
	}

	if ((*src).m_type == 2)	{
		i = (*src).m_head;
		for (size_t j = 0; j < sentence_length; j++)
		{
			if (m[i*sentence_length + j] && i != j) {
				(*src).m_result.insert(pair<string, float>(vItem[j], 1/fabs(m[i*sentence_length + j] - m[j*sentence_length + i])));
			}
		}
	}

	printf("\n");

	printf("var hours = [");
	for (size_t i = 0; i < sentence_length; i++)
		printf("'%s',", vItem[i].c_str());
	
	printf("];\n");
	printf("var days = [");
	for (size_t i = 0; i < sentence_length; i++)
		printf("'%s',", vItem[i].c_str());
	printf("];\n");

	printf("var data = [");
	for (size_t i = 0; i < sentence_length; i++)
	{
		for (size_t j = 0; j < sentence_length; j++)
		{
			printf("[%d,%d,%2.3f],\n", i, j, m[i*sentence_length + j]);
		}
	}
	printf("];\n");
	delete strUtf8;
}

void Neighbors_chunk(CQueryNeighbors *src) {
	printf("\nChunk distance of w2v:\n");
	char * strUtf8 = new char[MAXLEN];
	strcpy(strUtf8, src->m_head);
	ConvertGBKToUtf8(strUtf8);
	const long long N = 100;                  // number of closest words that will be shown
	char st1[MAXLEN];
	long long bi[100], d = 0;
	float dist, len, bestd[bestN], vec[MAXLEN];
	char *bestw[bestN];
	int a = 0, b = 0, c = 0, cn = 0;

	for (a = 0; a < N; a++) bestw[a] = (char *)malloc(MAXLEN * sizeof(char));
	for (a = 0; a < N; a++) bestd[a] = 0;
	for (a = 0; a < N; a++) bestw[a][0] = 0;
	for (a = 0; a < N; a++) bestw[a][0] = 0;
	cn++;
	for (a = 0; a < cn; a++) {
		for (b = 0; b < vocab_size; b++) if (!strcmp(vocab[b].word, strUtf8)) break;
		if (b == vocab_size) b = -1;
		bi[a] = b;
		if (b == -1) {
			printf("Out of dictionary word!\n");
			break;
		}
	}

	if (b == -1) { return; }
	for (a = 0; a < layer1_size; a++) vec[a] = 0;
	for (b = 0; b < cn; b++) {
		if (bi[b] == -1) continue;
		for (a = 0; a < layer1_size; a++) vec[a] += syn_chunk[a + bi[b] * layer1_size];
	}
	len = 0;
	for (a = 0; a < layer1_size; a++) len += vec[a] * vec[a];
	len = sqrt(len);
	for (a = 0; a < layer1_size; a++) vec[a] /= len;
	for (a = 0; a < N; a++) bestd[a] = -1;
	for (a = 0; a < N; a++) bestw[a][0] = 0;
	for (c = 0; c < vocab_size; c++) {
		a = 0;
		for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
		if (a == 1) continue;
		dist = 0;
		for (a = 0; a < layer1_size; a++) dist += vec[a] * syn_chunk[a + c * layer1_size];
		for (a = 0; a < N; a++) {
			if (dist > bestd[a]) {
				for (d = N - 1; d > a; d--) {
					bestd[d] = bestd[d - 1];
					strcpy(bestw[d], bestw[d - 1]);
					//res[d] = res[d - 1];
				}
				bestd[a] = dist;
				strcpy(bestw[a], vocab[c].word);
				//res[a] = c;
				break;
			}
		}
	}
	for (a = 0; a < N; a++) {
		strcpy(strUtf8, bestw[a]);
		ConvertUtf8ToGBK(strUtf8);
		printf("%50s\t\t%f\n", strUtf8, bestd[a]);
		(*src).m_result.insert(pair<string, float>(strUtf8, bestd[a]));
	}
	delete[] strUtf8;
	return;
}

void Semantic_chunk(CQuerySemantic *src) {
	printf("\nSemantic chunk of w2v:\n");
	char * strUtf8 = new char[MAXLEN];
	strcpy(strUtf8, src->m_first);
	ConvertGBKToUtf8(strUtf8);
	const long long N = 100;                  // number of closest words that will be shown
	char st1[MAXLEN];
	long long bi[100], d = 0;
	float dist, len, bestd[bestN], vec[MAXLEN];
	char *bestw[bestN];
	int a = 0, b = 0, c = 0, cn = 0;

	for (a = 0; a < N; a++) bestw[a] = (char *)malloc(MAXLEN * sizeof(char));
	for (a = 0; a < N; a++) bestd[a] = 0;
	for (a = 0; a < N; a++) bestw[a][0] = 0;
	for (a = 0; a < N; a++) bestw[a][0] = 0;
	cn++;
	for (a = 0; a < cn; a++) {
		for (b = 0; b < vocab_size; b++) if (!strcmp(vocab[b].word, strUtf8)) break;
		if (b == vocab_size) b = -1;
		bi[a] = b;
		if (b == -1) {
			printf("Out of dictionary word!\n");
			break;
		}
	}

	if (b == -1) { return; }
	for (a = 0; a < layer1_size; a++) vec[a] = 0;
	for (b = 0; b < cn; b++) {
		if (bi[b] == -1) continue;
		for (a = 0; a < layer1_size; a++) vec[a] += syn_chunk[a + bi[b] * layer1_size];
	}


	strcpy(strUtf8, src->m_second);
	ConvertGBKToUtf8(strUtf8);
	for (a = 0; a < cn; a++) {
		for (b = 0; b < vocab_size; b++) if (!strcmp(vocab[b].word, strUtf8)) break;
		if (b == vocab_size) b = -1;
		bi[a] = b;
		if (b == -1) {
			printf("Out of dictionary word!\n");
			break;
		}
	}
	for (b = 0; b < cn; b++) {
		if (bi[b] == -1) continue;
		for (a = 0; a < layer1_size; a++) vec[a] -= syn_chunk[a + bi[b] * layer1_size];
	}

	strcpy(strUtf8, src->m_input);
	ConvertGBKToUtf8(strUtf8);
	for (a = 0; a < cn; a++) {
		for (b = 0; b < vocab_size; b++) if (!strcmp(vocab[b].word, strUtf8)) break;
		if (b == vocab_size) b = -1;
		bi[a] = b;
		if (b == -1) {
			printf("Out of dictionary word!\n");
			break;
		}
	}
	for (b = 0; b < cn; b++) {
		if (bi[b] == -1) continue;
		for (a = 0; a < layer1_size; a++) vec[a] += syn_chunk[a + bi[b] * layer1_size];
	}


	len = 0;
	for (a = 0; a < layer1_size; a++) len += vec[a] * vec[a];
	len = sqrt(len);
	for (a = 0; a < layer1_size; a++) vec[a] /= len;
	for (a = 0; a < N; a++) bestd[a] = -1;
	for (a = 0; a < N; a++) bestw[a][0] = 0;
	for (c = 0; c < vocab_size; c++) {
		a = 0;
		for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
		if (a == 1) continue;
		dist = 0;
		for (a = 0; a < layer1_size; a++) dist += vec[a] * syn_chunk[a + c * layer1_size];
		for (a = 0; a < N; a++) {
			if (dist > bestd[a]) {
				for (d = N - 1; d > a; d--) {
					bestd[d] = bestd[d - 1];
					strcpy(bestw[d], bestw[d - 1]);
					//res[d] = res[d - 1];
				}
				bestd[a] = dist;
				strcpy(bestw[a], vocab[c].word);
				//res[a] = c;
				break;
			}
		}
	}
	for (a = 0; a < N; a++) {
		strcpy(strUtf8, bestw[a]);
		ConvertUtf8ToGBK(strUtf8);
		printf("%50s\t\t%f\n", strUtf8, bestd[a]);
		(*src).m_result.insert(pair<string, float>(strUtf8, bestd[a]));
	}
	delete[] strUtf8;
	return;
}

void IsActive(struct web_request_t* req, web_response_t* res)
{
	const char* key = NULL;
	const char* start = NULL;
	const char* fetch = NULL;
	web_put_status(res, SC_OK, "OK");
	web_put_header(res, "content-type", "text/plain");
	web_end_header(res);
	web_print(res, "ok");
}

typedef pair<string, float> PAIR;

bool cmp_by_value(const PAIR& lhs, const PAIR& rhs) {
	return lhs.second > rhs.second;
}

struct CmpByValue {
	bool operator()(const PAIR& lhs, const PAIR& rhs) {
		return lhs.second > rhs.second;
	}
};

void ChunkDistance(struct web_request_t* req, web_response_t* res) {
	//http://localhost:9091/chunk_distance?query=他们 对 历史 进行 反思&head=历史

	HANDLE hThread = NULL;
	DWORD ThreadID;
	CQuery query;

	const char* t = NULL;
	t = web_get_param(req, "query");
	if (t == NULL || *t == 0) return;
	strcpy(query.m_query, t);

	t = web_get_param(req, "head");
	if (t == NULL || *t == 0) return;
	query.m_head = atoi(t);

	t = web_get_param(req, "type");
	if (t == NULL || *t == 0) return;
	query.m_type = atoi(t);

	hThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)P_chunk, LPVOID(&query), 0, &ThreadID);
	WaitForSingleObject(hThread, INFINITE);

	web_put_status(res, SC_OK, "OK");
	web_put_header(res, "content-type", "text/plain");
	web_end_header(res);
	
	vector<PAIR> name_score_vec(query.m_result.begin(), query.m_result.end());
	//sort(name_score_vec.begin(), name_score_vec.end(), CmpByValue());
	sort(name_score_vec.begin(), name_score_vec.end(), cmp_by_value);

	t = new char[MAXLEN];
	for (int i = 0; i != name_score_vec.size(); ++i) {  
		web_print(res, name_score_vec[i].first.c_str());
		web_print(res, "\t");

		sprintf((char*)t, "%f", name_score_vec[i].second);
		web_print(res, t);

		web_print(res, "\n");
	}  
	delete t;
}

void ChunkNeighbors(struct web_request_t* req, web_response_t* res) {
	//http://localhost:9091/chunk_distance?query=他们 对 历史 进行 反思&head=历史

	HANDLE hThread = NULL;
	DWORD ThreadID;

	CQueryNeighbors query;

	const char* t = NULL;
	t = web_get_param(req, "head");
	if (t == NULL || *t == 0) return;
	strcpy(query.m_head, t);

	t = web_get_param(req, "dep");
	if (t == NULL || *t == 0) return;
	strcpy(query.m_dep, t);

	hThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)Neighbors_chunk, LPVOID(&query), 0, &ThreadID);
	WaitForSingleObject(hThread, INFINITE);

	web_put_status(res, SC_OK, "OK");
	web_put_header(res, "content-type", "text/plain");
	web_end_header(res);

	vector<PAIR> name_score_vec(query.m_result.begin(), query.m_result.end());
	//sort(name_score_vec.begin(), name_score_vec.end(), CmpByValue());
	sort(name_score_vec.begin(), name_score_vec.end(), cmp_by_value);

	t = new char[MAXLEN];
	for (int i = 0; i != name_score_vec.size(); ++i) {
		web_print(res, name_score_vec[i].first.c_str());
		web_print(res, "\t");

		sprintf((char*)t, "%f", name_score_vec[i].second);
		web_print(res, t);

		web_print(res, "\n");
	}
	delete t;
}

void ChunkSemantic(struct web_request_t* req, web_response_t* res) {
	//http://localhost:9091/chunk_distance?query=他们 对 历史 进行 反思&head=历史

	HANDLE hThread = NULL;
	DWORD ThreadID;

	CQuerySemantic query;

	const char* t = NULL;
	t = web_get_param(req, "first");
	if (t == NULL || *t == 0) return;
	strcpy(query.m_first, t);

	t = web_get_param(req, "second");
	if (t == NULL || *t == 0) return;
	strcpy(query.m_second, t);

	t = web_get_param(req, "input");
	if (t == NULL || *t == 0) return;
	strcpy(query.m_input, t);

	hThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)Semantic_chunk, LPVOID(&query), 0, &ThreadID);
	WaitForSingleObject(hThread, INFINITE);

	web_put_status(res, SC_OK, "OK");
	web_put_header(res, "content-type", "text/plain");
	web_end_header(res);

	vector<PAIR> name_score_vec(query.m_result.begin(), query.m_result.end());
	//sort(name_score_vec.begin(), name_score_vec.end(), CmpByValue());
	sort(name_score_vec.begin(), name_score_vec.end(), cmp_by_value);

	t = new char[MAXLEN];
	for (int i = 0; i != name_score_vec.size(); ++i) {
		web_print(res, name_score_vec[i].first.c_str());
		web_print(res, "\t");

		sprintf((char*)t, "%f", name_score_vec[i].second);
		web_print(res, t);

		web_print(res, "\n");
	}
	delete t;
}

int websvr_init(int nPort, int nMaxConn, char* psIP)
{
	/* int web_init(int port, int nconn,  int nSndBufSizeInKB, int nRcvBufSizeInKB) */
	if (web_init(nPort, nMaxConn, 8, 8)) {
		fprintf(stderr, "web_init: failed to init http server");
		return -1;
	}

	if (web_start()) {
		fprintf(stderr, "web_start: failed to start http server");
		return -1;
	}

	if (web_add_service("/isactive", IsActive)) {
		fprintf(stderr, "web_add_service: failed to add echo service");
		return -1;
	}
	if (web_add_service("/chunk_distance", ChunkDistance)) {
		fprintf(stderr, "web_add_service: failed to add echo service");
		return -1;
	}
	if (web_add_service("/chunk_neighbors", ChunkNeighbors)) {
		fprintf(stderr, "web_add_service: failed to add echo service");
		return -1;
	}
	if (web_add_service("/chunk_semantic", ChunkSemantic)) {
		fprintf(stderr, "web_add_service: failed to add echo service");
		return -1;
	}
	return 0;
}

bool Restart_Http(char* psPort)
{
	int port = atoi(psPort);

	printf("listening port:%d ...\n", port);
	fflush(stdin);

	if (websvr_init(port, 80, "127.0.0.1") == -1) {
		printf("start Failed!");
		return false;
	}

	char szInp[1024];
	printf("q to exit!\n");
	while (1) {
		scanf("%s", szInp);
		if (strcmp(szInp, "q") == 0)
			break;
	}
	web_final();
	return true;
}

bool init() {
	long long current_words = 0, words;
	FILE *fin;
	fin = fopen(chunk_file, "rb");
	if (fin == NULL) {
		printf("ERROR: chunk data file not found!\n");
		exit(1);
	}

	fscanf(fin, "%lld", &vocab_size);
	fscanf(fin, "%lld", &layer1_size);
	fscanf(fin, "%lld", &chunk_window);
	fscanf(fin, "%lld", &ontology_size);

	vocab = (struct vocab_word *)calloc(vocab_size, sizeof(struct vocab_word));
	posix_memalign((void **)&syn_chunk, 128, (long long)vocab_size * layer1_size * sizeof(real));
	posix_memalign((void **)&syn_internode, 128, (long long)(vocab_size * 2 + ontology_size) * layer1_size * sizeof(real));
	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
	
	for (size_t a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	/*
	for (size_t a = 0; a < vocab_size; a++) {
		vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}
	*/
	char word[MAX_STRING]; 

	for (size_t i = 0; i < vocab_size; i++)
	{
		ReadWord(word, fin);
		if (feof(fin)) break;
		current_words++;
		if ((debug_mode > 1) && (current_words % 100000 == 0)) {
			printf("%lldK%c", current_words / 1000, 13);
			fflush(stdout);
		}
		AddWordToVocab(word, i);

		fread(&vocab[i].cn, sizeof(long long), 1, fin);
		fread(&vocab[i].codelen, sizeof(char), 1, fin);

		vocab[i].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));	//////????????????????
		vocab[i].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));	//////????????????????

		for (size_t a = 0; a < vocab[i].codelen; a++) 
			fread(&vocab[i].point[a], sizeof(int), 1, fin);
		for (size_t a = 0; a < vocab[i].codelen; a++) fread(&vocab[i].code[a], sizeof(char), 1, fin);
		for (size_t a = 0; a < layer1_size; a++) fread(&syn_chunk[i * layer1_size + a], sizeof(real), 1, fin);
	}

	for (size_t i = 0; i < vocab_size * 2 + ontology_size; i++)
		for (size_t a = 0; a < layer1_size; a++) 
			fread(&syn_internode[i * layer1_size + a], sizeof(float), 1, fin);

	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	for (int i = 0; i < EXP_TABLE_SIZE; i++) {
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}

	return true;
}

int _tmain(int argc, char* argv[])
{
	port[0] = 0;
	chunk_file[0] = 0;

	int i = 0;
	if ((i = ArgPos((char *)"-port", argc, argv)) > 0) strcpy(port, argv[i + 1]);
	if ((i = ArgPos((char *)"-chunk-file", argc, argv)) > 0) strcpy(chunk_file, argv[i + 1]);
	
	if (!init()) return 0;
	Restart_Http(port);
    return 0;
}


//-----------TOOLS-----------
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
// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
	int a = 0, ch;
	while (!feof(fin)) {
		ch = fgetc(fin);
		if (ch == 13) continue;
		if ((ch == ' ') && (a > 0)) break;
		word[a] = ch;
		a++;
		if (a >= MAX_STRING - 1) 
			a--;   // Truncate too long words
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

// Adds a word to the vocabulary
int AddWordToVocab(char *word, long long vocab_pos) {
	unsigned int hash, length = strlen(word) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	vocab[vocab_pos].word = (char *)calloc(length, sizeof(char));
	strcpy(vocab[vocab_pos].word, word);
	vocab[vocab_pos].cn = 0;
	vocab_pos++;
	// Reallocate memory if needed
	if (vocab_pos + 2 >= vocab_max_size) {
		vocab_max_size += 1000;
		vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
	}
	hash = GetWordHash(word);
	while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
	vocab_hash[hash] = vocab_pos - 1;
	return vocab_pos - 1;
}

void ConvertUtf8ToGBK(char* &strUtf8)
{
	int len = MultiByteToWideChar(CP_UTF8, 0, (LPCCH)strUtf8, -1, NULL, 0);
	wchar_t * wszGBK = new wchar_t[len];
	memset(wszGBK, 0, len);
	MultiByteToWideChar(CP_UTF8, 0, (LPCCH)strUtf8, -1, wszGBK, len);
	len = WideCharToMultiByte(CP_ACP, 0, wszGBK, -1, NULL, 0, NULL, NULL);
	char *szGBK = new char[len + 1];
	memset(szGBK, 0, len + 1);
	WideCharToMultiByte(CP_ACP, 0, wszGBK, -1, szGBK, len, NULL, NULL);
	strcpy(strUtf8, szGBK);
	delete[] szGBK;
	delete[] wszGBK;
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


void SplitString(const char* psStr, const char* psSeparator, vector<string> & vItem)
{
	char szTmp[MAXLEN];
	strcpy(szTmp, psStr);
	vItem.clear();
	char* psTmp;
	psTmp = strtok(szTmp, psSeparator);
	while (psTmp != NULL) {
		if (strlen(psTmp)<MAXLEN && strlen(psTmp)>0)
			vItem.push_back(psTmp);
		psTmp = strtok(NULL, psSeparator);
	}
}

string JoinString(const char* psSeparator, const vector<string> vItem)
{
	string res;
	if (vItem.size() < 1) return false;
	int nres = 0;
	int a = 0;
	for (a = 0; a < vItem.size() - 1; a++) {
		res.append(vItem[a].c_str());
		res.append(psSeparator);
	}
	res.append(vItem[a].c_str());
	return res;
}