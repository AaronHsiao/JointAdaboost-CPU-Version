#include <cstdio>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <ctime>
#include <algorithm>

#include "compute.hpp"
using namespace std;

//自定結構 每做完一次WeakLeaner 會產生一組WeakLeanerOutput的結構 再存到F二維陣列之中
struct WeakLeanerOutput
{
	public:
		float theta;
		float polarity;
		float final_error;

		WeakLeanerOutput()
		{
			theta = 0;
			polarity = 0;
			final_error = 0;
		}
};

struct Model
{
	public:
		float** bestTable;
		float alpha;
		float x;	//X軸的Feautre編號
		float y;	//Y軸的Feautre編號

		~Model()
		{
			for (int i = 0; i < 4; i++)
			{
				delete[] bestTable[i];
			}
		}
};


void WeakLearn(float[][15], float[][14], float[], float[], int, int, int, int[][2], float*, float**, int**);
void AdaBoostTrain(float pf[][15], float nf[][14], int pf_sn, int nf_sn, int fn, int times, int list[][2]);
void AdaBoostTest(float[][5], int , int );
float MyRound(float);
string PorN(float);

Compute compute("WeakLearn", CL_DEVICE_TYPE_GPU);

const int times = 5;	//訓練次數
float F[times][4];	//用二維矩陣 存放每次訓練完之結果 4分別代表著 1. selectif(選到的Feature) 2. polarity(右邊是正or負資料) 3. error(錯誤率) 4. alpha值 

//最終的模型(times張表)
Model* model;

float **q_Map;

int main()
{
	model = new Model[times];

	////float *err_WeakLearn = weakLearn(pf[0], nf[0], pw[0], nw[0], sizeof(pf) / sizeof(pf[0]), sizeof(nf) / sizeof(nf[0]));

	//char file_Train_PF1[] = "G:\\Train_PF1.txt";   //2429*2101
	//char file_Train_NF1[] = "G:\\Train_NF1.txt";   //4548*2101
	//char file_Test_PF1[] = "G:\\Test_PF1.txt";	   //472*2101
	//char file_Test_NF1[] = "G:\\Test_NF1.txt";	   //23573*2101

	//const int row_Train_PF1 = 2101;
	//const int column_Train_PF1 = 2429;

	//auto arr_Train_PF1 = new float[row_Train_PF1][column_Train_PF1];

	//////動態配置二維矩陣 否則會StackOverFlow
	////arr_Train_PF1 = new float*[row_Train_PF1];
	////for (int i = 0; i<row_Train_PF1; i++)
	////	arr_Train_PF1[i] = new float[column_Train_PF1];

	//fstream fp1;
	//char line1[256];

	//fp1.open(file_Train_PF1, ios::in);//開啟檔案
	//if (!fp1){//如果開啟檔案失敗，fp為0；成功，fp為非0
	//	cout << "Fail to open file: " << file_Train_PF1 << endl;
	//}

	//int i1 = 0;
	//int j1 = 0;

	//while (fp1.getline(line1, sizeof(line1), '\t'))
	//{
	//	if (i1 == column_Train_PF1)
	//		break;

	//	arr_Train_PF1[j1][i1] = atof(line1);

	//	j1++;

	//	if (j1 == row_Train_PF1)
	//	{
	//		j1 = 0;
	//		i1++;
	//	}
	//}
	////cout << arr_Train_PF1[0][0] << endl;
	////cout << arr_Train_PF1[0][2428] << endl;
	////cout << arr_Train_PF1[2100][0] << endl;
	////cout << arr_Train_PF1[2100][2428] << endl;

	//fp1.close();//關閉檔案

	//const int row_Train_NF1 = 2101;
	//const int column_Train_NF1 = 4548;
	//auto arr_Train_NF1 = new float[row_Train_NF1][column_Train_NF1];

	////動態配置二維矩陣 否則會StackOverFlow
	////arr_Train_NF1 = new float*[row_Train_NF1];
	////for (int i = 0; i<row_Train_NF1; i++)
	////	arr_Train_NF1[i] = new float[column_Train_NF1];

	//fstream fp2;
	//char line2[256];

	//fp2.open(file_Train_NF1, ios::in);//開啟檔案
	//if (!fp2){//如果開啟檔案失敗，fp為0；成功，fp為非0
	//	cout << "Fail to open file: " << file_Train_NF1 << endl;
	//}

	//int i2 = 0;
	//int j2 = 0;

	//while (fp2.getline(line2, sizeof(line2), '\t'))
	//{
	//	if (i2 == column_Train_NF1)
	//		break;

	//	arr_Train_NF1[j2][i2] = atof(line2);

	//	j2++;

	//	if (j2 == row_Train_NF1)
	//	{
	//		j2 = 0;
	//		i2++;
	//	}
	//}

	//fp2.close();//關閉檔案

	////const int row_Test_PF1 = 472;
	////const int column_Test_PF1 = 2101;
	////float **arr_Test_PF1;

	//////動態配置二維矩陣 否則會StackOverFlow
	////arr_Test_PF1 = new float*[row_Test_PF1];
	////for (int i = 0; i<row_Test_PF1; i++)
	////	arr_Test_PF1[i] = new float[column_Test_PF1];

	////fstream fp3;
	////char line3[128];

	////fp3.open(file_Test_PF1, ios::in);//開啟檔案
	////if (!fp3){//如果開啟檔案失敗，fp為0；成功，fp為非0
	////	cout << "Fail to open file: " << file_Test_PF1 << endl;
	////}

	////int i3 = 0;
	////int j3 = 0;

	////while (fp3.getline(line3, sizeof(line3), '\t'))
	////{
	////	if (i3 == row_Test_PF1)
	////		break;

	////	arr_Test_PF1[i3][j3] = atof(line3);

	////	j3++;

	////	if (j3 == column_Test_PF1)
	////	{
	////		j3 = 0;
	////		i3++;
	////	}
	////}

	////fp3.close();//關閉檔案

	////const int row_Test_NF1 = 23573;
	//const int column_Test_NF1 = 2101;
	////float **arr_Test_NF1;

	//////動態配置二維矩陣 否則會StackOverFlow
	////arr_Test_NF1 = new float*[row_Test_NF1];
	////for (int i = 0; i<row_Test_NF1; i++)
	////	arr_Test_NF1[i] = new float[column_Test_NF1];

	////fstream fp4;
	////char line4[128];

	////fp4.open(file_Test_NF1, ios::in);//開啟檔案
	////if (!fp4){//如果開啟檔案失敗，fp為0；成功，fp為非0
	////	cout << "Fail to open file: " << file_Test_NF1 << endl;
	////}

	////int i4 = 0;
	////int j4 = 0;

	////while (fp4.getline(line4, sizeof(line4), '\t'))
	////{
	////	if (i4 == row_Test_NF1)
	////		break;

	////	arr_Test_NF1[i4][j4] = atof(line4);

	////	j4++;

	////	if (j4 == column_Test_NF1)
	////	{
	////		j4 = 0;
	////		i4++;
	////	}
	////}

	////fp4.close();//關閉檔案

	const int feature_Size = 3;	//特徵數量
	const int pf_sn_Train = 15;	//正臉Sample數量
	const int nf_sn_Train = 14;	//負臉Sample數量


	/*		q_Map示意圖
	Q1	Q2	Q3
	1
	2
	3
	.
	.
	.
	fn
	*/

	//四分位數的Map 二維陣列
	q_Map = new float*[feature_Size];

	for (int x = 0; x < feature_Size; x++)
		q_Map[x] = new float[3];




	//PF Sample Data
	float TrainPF[feature_Size][pf_sn_Train] = { {1, 2, 7, 7, 7, 4, 5, 8, 6, 5, 9, 10, 3, 6, 8},
													{7, 9, 3, 4, 6, 2, 2, 0, 1, 3, 4, 5, 7, 10, 2 },
													{3, 6, 6, 6, 2, 4, 3, 1, 5, 8, 9, 5, 7, 9, 3 } };

	//NF Sample Data
	float TrainNF[feature_Size][nf_sn_Train] = { { 9, 1, 3, 4, 6, 2, 2, 2, 2, 7, 0, 8, 1, 3},
													{ 7, 7, 4, 6, 3, 5, 6, 3, 2, 8, 9, 3, 1, 2},
													{ 8, 5, 0, 10, 10, 2, 4, 1, 3, 5, 7, 2, 0, 9} };

	//list[][] C3取2 示意圖
	/*
		[0][1]
		[0][2]
		[1][2]
	*/

	const int cn2 = feature_Size * (feature_Size - 1) / 2;

	//CN取2的清單
	int list[cn2][2];

	//處理CN取2的實作 結果存於list
	int start = 0;
	int start_Nei = start + 1;
	for (int i = 0; i < cn2; ++i)
	{
		list[i][0] = start;
		list[i][1] = start_Nei;

		if (start_Nei + 1 >= feature_Size)
		{
			start++;
			start_Nei = start + 1;
		}
		else
		{
			start_Nei++;
		}
	}

	/*檢查CN取2的結果
	for (int j = 0; j < times; ++j)
	{
		cout << list[j][0] << ", " << list[j][1] << endl;
	}
	*/

	clock_t begin = clock();
	
	//參數說明: 正資料pointer, 负資料pointer, 正資料sample數, 负資料sample數, 特徵數量, 訓練次數, CN取2的清單pointer
	AdaBoostTrain(TrainPF, TrainNF, pf_sn_Train, nf_sn_Train, feature_Size, times, list);

	const int pf_sn_Test = 5;	//正臉Sample數量
	const int nf_sn_Test = 4;	//負臉Sample數量

	//NF Sample Data
	float TestPF[feature_Size][pf_sn_Test] = { { 3, 1, 2, 4, 5 },
												{ 0, 1, 7, 3, 5 },
												{ 7, 1, 6, 2, 2 }, };

	int alphaSum;
	AdaBoostTest(TestPF, pf_sn_Test, feature_Size);

	//int TP = AdaBoostTest(arr_Test_PF1[0], row_Test_PF1, column_Test_NF1);
	//printf("%f/n", TP / row_Test_PF1);
	//int FP = row_Test_NF1-AdaBoostTest(arr_Test_NF1[0], row_Test_NF1, column_Test_NF1);
	//printf("%f/n", FP / row_Test_NF1);
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Time: " << elapsed_secs << " seconds!!!!" << endl;

	//釋放記憶體
	for (int x = 0; x < feature_Size; x++)
	{
		delete[] q_Map[x];
	}
	delete[] q_Map;

	delete[] model;


	//for (int i = 0; i < row_Train_PF1; i++)
	//{
	//	delete[] arr_Train_PF1[i];
	//}

	//delete[] arr_Train_PF1;
	//delete[] arr_Train_NF1;

	
}

//參數說明: 正資料pointer, 负資料pointer, 正資料sample數, 负資料sample數, 特徵數量, 訓練次數, CN取2的清單pointer
void AdaBoostTrain(float pf[][15], float nf[][14], int pf_sn, int nf_sn, int fn, int times, int list[][2])
{
	//總資料個數total_SampleNumber
	int total_sn = pf_sn + nf_sn;
	
	//正副資料結合成一大條暫時的陣列
	float *temp_Range = new float[total_sn];

	//算四分位
	for (int g = 0; g < fn; g++)
	{
		//讀正資料進來
		for (int z = 0; z < pf_sn; z++)
			temp_Range[z] = pf[g][z];

		//讀负資料進來
		for (int z = 0; z < nf_sn; z++)
			temp_Range[z + pf_sn] = nf[g][z];

		/*
		cout << "檢查還沒排序過的temp_Range";
		for (int z = 0; z < total_sn; z++)
			cout << "" << temp_Range[z] << ", ";
		cout << "\n";
		*/

		//利用內建Library排序瞜 傳入頭尾即可
		sort(temp_Range, temp_Range+total_sn);

		/*
		cout << "檢查排序過的temp_Range";
		for (int z = 0; z < total_sn; z++)
			cout << " " << temp_Range[z] << ", ";
		cout << "\n";
		system("pause");
		*/

		float q1_Index = (total_sn + 1) / 4.0;
		float q2_Index = (total_sn + 1) / 2.0;
		float q3_Index = (total_sn + 1) / 4.0 * 3.0;

		//cout << "檢查Q1、Q2、Q3\n";
		//cout << q1_Index << ", " << q2_Index << ", " << q3_Index << "\n";

		//內插法處理 7.5這種情形 (7+8)/2
		if ((total_sn + 1) % 4 == 0)
		{
			q_Map[g][0] = temp_Range[(int)q1_Index-1];			
			q_Map[g][2] = temp_Range[(int)q3_Index-1];
		}
		else
		{
			q_Map[g][0] = (temp_Range[(int)q1_Index-1] + temp_Range[(int)q1_Index])/2;		
			q_Map[g][2] = (temp_Range[(int)q3_Index-1] + temp_Range[(int)q3_Index])/2;
		}

		//Q2除以四後還要乘二，獨立處理
		if ((total_sn + 1) % 2 == 0)
		{
			q_Map[g][1] = temp_Range[(int)q2_Index-1];
		}
		else
		{
			q_Map[g][1] = (temp_Range[(int)q2_Index-1] + temp_Range[(int)q2_Index]) / 2;
		}

		//cout << "檢查qIndex\n";
		//cout << (int)q1_Index << ", " << (int)q2_Index << ", " << (int)q3_Index << "\n";
	}

	delete[] temp_Range;

	/*
	cout << "檢查q_Map" << "\n";
	for (int g = 0; g < fn; g++)
	{
		cout << q << ": " << q_Map[g][0] << ", " << q_Map[g][1] << ", " << q_Map[g][2] << "\n";
	}
	system("pause");
	*/

	int cn2 = fn*(fn - 1) / 2;		//cn取2

	float *pw = new float[pf_sn];	//正資料權重
	float *nw = new float[nf_sn];	//负資料權重
	float wsum = 0;

	//初使化正负臉權重 個資料權重一開始相同

	for (int i = 0; i < pf_sn; i++)		
		pw[i] = 1.0 / (total_sn);
		//pw[i] = 0.5 / pf_sn;

	for (int i = 0; i < nf_sn; i++)
		nw[i] = 1.0 / (total_sn);
		//nw[i] = 0.5 / nf_sn;

	for (int i = 0; i < pf_sn; i++)
		wsum = wsum + pw[i];

	for (int i = 0; i < nf_sn; i++)
		wsum = wsum + nw[i];

	for (int i = 0; i < pf_sn; i++)
		pw[i] /= wsum;

	for (int i = 0; i < nf_sn; i++)
	{
		nw[i] /= wsum;
	}

	/*
	cout << "檢查 pw 正資料權重";
	for (int i = 0; i < pf_sn; i++)
		cout << pw[i] << ", ";

	cout << "\n";
	system("pause");

	cout << "檢查 nw 负資料權重";
	for (int i = 0; i < nf_sn; i++)
		cout << nw[i] << ", ";

	cout << "總權重 wsum \n";
	cout << "wsum: " << wsum;

	system("pause");
	*/

	//float ret[2101][3];

	//OpenCL, 多傳一個List進去吧
	//int pf_shape[2] = { fn, pf_sn };
	//int nf_shape[2] = { fn, nf_sn };

	//compute.set_buffer((float *)pf, fn * pf_sn*sizeof(float));
	//compute.set_buffer((float *)nf, fn * nf_sn*sizeof(float));

	//compute.set_buffer((float *)pw, pf_sn*sizeof(float));
	//compute.set_buffer((float *)nw, nf_sn*sizeof(float));

	//compute.set_buffer((int *)pf_shape, 2 * sizeof(int));
	//compute.set_buffer((int *)nf_shape, 2 * sizeof(int));

	//compute.set_buffer(1);

	//compute.set_ret_buffer((float *)ret, fn * 3 * sizeof(float));



	//做times次的WeakLearner，換句話說，在此之上的程式碼只會執行一次
	for (int i = 0; i < times; i++)
	{
		wsum = 0;

		//調整權重
		for (int i = 0; i < pf_sn; i++)
			wsum = wsum + pw[i];

		for (int i = 0; i < nf_sn; i++)
			wsum = wsum + nw[i];

		for (int i = 0; i < pf_sn; i++)
			pw[i] /= wsum;

		for (int i = 0; i < nf_sn; i++)
			nw[i] /= wsum;

		/*
		cout << "檢查 pw 正資料權重";
		for (int i = 0; i < pf_sn; i++)
		cout << pw[i] << ", ";

		cout << "\n";
		system("pause");

		cout << "檢查 nw 负資料權重";
		for (int i = 0; i < nf_sn; i++)
		cout << nw[i] << ", ";

		cout << "總權重 wsum \n";
		cout << "wsum: " << wsum;
		system("pause");
		*/

		//輸出的陣列 存WeakLearn跑完輸出的各Feature錯誤率 
		float *err_WeakLearn = new float[cn2];
		
		//每一個資料點的座標 一對一的記錄 所以有正臉數+副臉數
		//sn_XY會傳入WeakLearn進行計算
		int **sn_XY = new int*[total_sn];
		for (int k = 0; k < total_sn; k++)
			sn_XY[k] = new int[2];

		//compute.reset_buffer(2, pw);
		//compute.reset_buffer(3, nw);

		//幾個Kernel在跑
		//compute.run(2101);
		
		/*
			參數說明 正資料pointer, 负資料pointer, 正資料權重, 负資料權重, 正資料個數, 负資料個數,
			特徵數量, 組合式特徵清單, 回傳陣列, q_Map pointer
		*/
		WeakLearn(pf, nf,pw, nw, pf_sn, nf_sn, fn, list, err_WeakLearn, q_Map, sn_XY);

		//float theta = 0; 
		//float polarity = 1;
		float error = 1;
		int selectif = -1;
		float beta;

		for (int k = 0; k < cn2; k++)
		{
			if (err_WeakLearn[k] < error)
			{
				error = err_WeakLearn[k];
				selectif = k;
			}

			cout << "\n";
			cout << "err_WeakLearn[" << k << "]= " << err_WeakLearn[k];
			system("pause");
		}



		//Rebuild the error map, given k we know features


		//if (error > err_WeakLearn[i].final_error)
		//{
		//	error = err_WeakLearn[i].final_error;
		//	polarity = err_WeakLearn[i].polarity;
		//	theta = err_WeakLearn[i].theta;
		//	selectif = i;	//最好的那一"行"特徵 (從0開始算，跟Matlab誤差1)
		//}


		//printf("%f, %f, %f, %f\n", err_WeakLearn[i].theta, err_WeakLearn[i].polarity, err_WeakLearn[i].final_error, selectif);
		
		//公式可知 beta一定是越來越小
		beta = error / (1 - error);


		//WeakLearn完成後的16宮格
		float **bestTable = new float*[4];
		for (int b = 0; b < 4; b++)
			bestTable[b] = new float[4];

		int indX = list[selectif][0];
		int indY = list[selectif][1];

		model[i].x = indX;
		model[i].y = indY;

		//投票瞜
		for (int g = 0; g < total_sn; g++)
		{
			if (g < pf_sn)
				bestTable[sn_XY[g][0]][sn_XY[g][1]] += pw[g];
			else
				bestTable[sn_XY[g][0]][sn_XY[g][1]] -= nw[g - pf_sn];
		}

		//正臉調權重
		for (int i = 0; i < pf_sn; i++)
		{			
			if (bestTable[sn_XY[i][0]][sn_XY[i][1]] >= 0)
			{
				pw[i] = pw[i] * beta;
			}
		}

		//負臉調權重
		for (int i = pf_sn; i < nf_sn+pf_sn; i++)
		{
			if (bestTable[sn_XY[i][0]][sn_XY[i][1]] < 0)
			{
				nw[i-pf_sn] = nw[i-pf_sn] * beta;
			}
		}

		//每一次訓練完產出的Table 紀錄在Model之中
		model[i].bestTable = bestTable;

		//cout << "pwpwpwpwpwpwp,,,,";
		//for (int i = 0; i < pf_sn; i++)
		//{
		//	cout << pw[i] << ", ";
		//}

		//cout << "\n";
		//system("pause");

		//cout << "nwnwnw,,,,";
		//for (int i = 0; i < nf_sn; i++)
		//{
		//	cout << nw[i] << ", ";
		//}

		//cout << "\n\n";
		//cout << "wsum: " << wsum;
		//system("pause");

		//printf("i=%d\n", i);

		//F[i][0] = selectif;
		//F[i][1] = polarity;
		//F[i][2] = theta;
		//F[i][3] = log(1 / beta);

		//把alpha值 四捨五入至小數第四位
		F[i][3] = MyRound(F[i][3]);

		model[i].alpha = log(1.0 / beta);

		//cout << "beta: " << beta << "\n";
		//cout << "alpha" << log(1.0 / beta) << "\n";

		/*if (i % 100 == 0)*/
		printf("[%d+%d] , %d, %f\n", list[selectif][0], list[selectif][1], selectif + 1, model[i].alpha);
		for (int q = 0; q < 4; q++)
		{
			//cout << PorN(bestTable[q][0])
			//	<< "|" << PorN(bestTable[q][1])
			//	<< "|" << PorN(bestTable[q][2])
			//	<< "|" << PorN(bestTable[q][3]) << "\n";
		}

		//釋放記憶體
		for (int i = 0; i < total_sn; i++)
			delete[] sn_XY[i];

		delete[] sn_XY;
	}



	delete[] pw;
	delete[] nw;
}


//自製小數點四捨五入至小數第四位
float MyRound(float number)
{
	float f = floor(number * 10000 + 0.5) / 10000;
	return f;
}

string PorN(float num)
{
	if (num >= 0)
		return "+";
	else
		return "-";
}

void AdaBoostTest(float data[][5], int data_sn, int fn)
{
	int *score = new int[data_sn];
	float alphaSum = 0;

	for (int i = 0; i < times; i++)
	{
		alphaSum += model[i].alpha;

		int x = model[i].x;
		int y = model[i].y;

		//sn_XY會傳入WeakLearn進行計算
		int **test_XY = new int*[data_sn];
		for (int k = 0; k < data_sn; k++)
			test_XY[k] = new int[2];

		for (int j = 0; j < data_sn; j++)
		{
			if (data[x][j] < q_Map[x][0])
				test_XY[j][0] = 0;
			else if (data[x][j] < q_Map[x][1])
				test_XY[j][0] = 1;
			else if (data[x][j] < q_Map[x][2])
				test_XY[j][0] = 2;
			else
				test_XY[j][0] = 3;

			if (data[y][j] < q_Map[y][0])
				test_XY[j][1] = 0;
			else if (data[y][j] < q_Map[y][1])
				test_XY[j][1] = 1;
			else if (data[y][j] < q_Map[y][2])
				test_XY[j][1] = 2;
			else
				test_XY[j][1] = 3;

			if (model[i].bestTable[test_XY[j][0]][test_XY[j][1]] >= 0)
				score[j] += model[i].alpha;
		}
	}

	cout << "\nalphaSum: " << alphaSum << "\n";
	system("pause");

	//float *predit = new float[data_sn];
	//int count = 0;
	//for (int i = 0; i < data_sn; i++)
	//{
	//	for (int j = 0; j < data_fn; j++)
	//	{
	//		for (int k = 0; k < sizeof(F) / sizeof(F[0]); k++)
	//		{
	//			if (F[k][1] == 1)
	//			{
	//				if (data[i*(data_sn - 1) + j] >= F[k][2])
	//				{
	//					predit[i] = predit[i] + F[k][3];
	//				}
	//			}
	//			else
	//			{
	//				if (data[i*(data_sn - 1) + j] < F[k][2])
	//				{
	//					predit[i] = predit[i] + F[k][3];
	//				}
	//			}
	//		}
	//	}
	//	if (predit[i] > 0.5)
	//	{
	//		count++;
	//	}
	//}
	//return count;
}

/*
參數說明 正資料pointer, 负資料pointer, 正資料權重, 负資料權重,
			正資料個數, 负資料個數, 特徵數量, 組合式特徵清單, 回傳陣列, q_Map pointer 
*/
void WeakLearn(float pf[][15], float nf[][14], float pw[], float nw[], 
	int pf_sn, int nf_sn, int fn, int list[][2], float* return_Matrix, float** q_Map, int** sn_XY)
{
	/*
	cout << "檢查 pw 正資料權重:";
	for (int i = 0; i < pf_sn; i++)
	{
		cout << pw[i] << ", ";
	}
	cout << "\n";
	system("pause");

	cout << "檢查 nw 负資料權重:";
	for (int i = 0; i < nf_sn; i++)
	{
		cout << nw[i] << ", ";
	}

	cout << "\n";
	system("pause");
	*/

	int total_sn = pf_sn + nf_sn;		//總資料個數 = 正資料+负資料
	int cn2 = fn*(fn - 1) / 2;			//CN取2個特徵

	//h(header)控制最外層的迴圈
	for (int h = 0; h < cn2; ++h)
	{	
		/*  
			list[][] C3取2 示意圖

			 X  Y
			[0][1]
			[0][2]
			[1][2]
		*/

		int adaboost_XAxis = list[h][0];	//JointAdaboost 16宮格之X軸 代表第幾個Feature ex: 0 => RSI
		int adaboost_YAxis = list[h][1];	//JointAdaboost 16宮格之Y軸 代表第幾個Feature ex: 1 => KD

		//根據座標決定取出pf的某整條特徵 進行Joint	
		float* pf_X = pf[adaboost_XAxis];
		float* pf_Y = pf[adaboost_YAxis];
		
		//根據座標決定取出nf的某整條特徵 進行Joint	
		float* nf_X = nf[adaboost_XAxis];
		float* nf_Y = nf[adaboost_YAxis];
		
		//16宮格 兩軸都混合正负資料 取出最大最小
		float min_X = pf_X[0];
		float max_X = pf_X[0];
		float min_Y = pf_Y[0];
		float max_Y = pf_Y[0];

		//float max = , min = 20000000000000, error = 1, theta = 0, polarity = 1;
		
		//正資料裡面 找最大 最小值
		for (int i = 0; i < pf_sn; i++)
		{
			if (pf_X[i] > max_X)
			{
				max_X = pf_X[i];
			}
			if (pf_X[i] < min_X)
			{
				min_X = pf_X[i];
			}

			if (pf_Y[i] > max_Y)
			{
				max_Y = pf_Y[i];
			}
			if (pf_Y[i] < min_Y)
			{
				min_Y = pf_Y[i];
			}
		}

		//负資料 接續剛剛正資料 繼續找最大最小值
		for (int i = 0; i < nf_sn; i++)
		{
			if (nf_X[i] > max_X)
			{
				max_X = nf_X[i];
			}
			if (nf_X[i] < min_X)
			{
				min_X = nf_X[i];
			}

			if (nf_Y[i] > max_Y)
			{
				max_Y = nf_Y[i];
			}
			if (nf_Y[i] < min_Y)
			{
				min_Y = nf_Y[i];
			}
		}

		//seatTable計票用的16宮格!!!!
		float seatTable[4][4] = { { 0, 0, 0, 0 },
									{ 0, 0, 0, 0 }, 
									{ 0, 0, 0, 0 }, 
									{ 0, 0, 0, 0 }};

		//seatTable16宮格延伸兩個屬性去紀錄 +-總合 (P=Postive, N=Negative)
		float seatTable_PN[4][4][2] = { { { 0, 0}, { 0, 0}, { 0, 0}, { 0, 0} },
										{ { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } },
										{ { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } },
										{ { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, };


		/*	
				SampleNumber_XY 對每筆資料選出最好的XY值
						X	Y
				1		   
				2		   
				.
				.
				.
		   pf_sn+fn_sn
		*/

		//先把正資料 丟到16宮格 看看座落於哪個座標
		for (int z = 0; z < pf_sn; z++)
		{		
			if (pf_X[z] < q_Map[adaboost_XAxis][0])
				sn_XY[z][0] = 0;
			else if (pf_X[z] < q_Map[adaboost_XAxis][1])
				sn_XY[z][0] = 1;
			else if (pf_X[z] < q_Map[adaboost_XAxis][2])
				sn_XY[z][0] = 2;
			else
				sn_XY[z][0] = 3;

			if (pf_Y[z] < q_Map[adaboost_YAxis][0])
				sn_XY[z][1] = 0;
			else if (pf_Y[z] < q_Map[adaboost_YAxis][1])
				sn_XY[z][1] = 1;
			else if (pf_Y[z] < q_Map[adaboost_YAxis][2])
				sn_XY[z][1] = 2;
			else
				sn_XY[z][1] = 3;
		}

		//接著把负資料也丟入16宮格 找出座標
		for (int z = pf_sn; z < total_sn; z++)
		{
			if (nf_X[z - pf_sn] < q_Map[adaboost_XAxis][0])
				sn_XY[z][0] = 0;
			else if (nf_X[z - pf_sn] < q_Map[adaboost_XAxis][1])
				sn_XY[z][0] = 1;
			else if (nf_X[z - pf_sn] < q_Map[adaboost_XAxis][2])
				sn_XY[z][0] = 2;
			else
				sn_XY[z][0] = 3;

			if (nf_Y[z - pf_sn] < q_Map[adaboost_YAxis][0])
				sn_XY[z][1] = 0;
			else if (nf_Y[z - pf_sn] < q_Map[adaboost_YAxis][1])
				sn_XY[z][1] = 1;
			else if (nf_Y[z - pf_sn] < q_Map[adaboost_YAxis][2])
				sn_XY[z][1] = 2;
			else
				sn_XY[z][1] = 3;

			//cout << "nf_X: " << nf_X[z - pf_sn] <<", nf_Y: " << nf_Y[z - pf_sn] << "\n";
			//system("pause");
		}

		/*
		cout << "q_Map" << "\n";
		for (int x = 0; x < fn; x++)
			cout << q_Map[x][0] << ", " << q_Map[x][1] << ", " << q_Map[x][2] << "\n";

		cout << "sn_XY[][0] sn_XY[][0]";
		for (int x = 0; x < total_sn; x++)
			cout << sn_XY[x][0] << sn_XY[x][1] << "\n";
		system("pause");

		cout << "pw\n";
		for (int j = 0; j < pf_sn; j++)
			cout << pw[j] << ", ";

		cout << "\n";

		cout << "nw\n";
		for (int j = 0; j < nf_sn; j++)
			cout << nw[j] << ", ";

		system("pause");
		*/

		return_Matrix[h] = 0;

		//有了每個資料點所在的XY座標 就可以進行投票(加減權重)
		for (int g = 0; g < total_sn; g++)
		{
			if (g < pf_sn)
			{
				seatTable[sn_XY[g][0]][sn_XY[g][1]] += pw[g];		//直接就是存每一格總結果
				seatTable_PN[sn_XY[g][0]][sn_XY[g][1]][0] += pw[g]; //統記每一格的正權重
				//return_Matrix[h] += pw[g];
			}
				
			else
			{
				seatTable[sn_XY[g][0]][sn_XY[g][1]] -= nw[g - pf_sn];			//直接就是存每一格總結果
				seatTable_PN[sn_XY[g][0]][sn_XY[g][1]][1] += nw[g - pf_sn];		//統記每一格的负權重(正值)
				//return_Matrix[h] -= nw[g - pf_sn];
			}

			//cout << "[" << sn_XY[g][0] << "]["<< sn_XY[g][1] << "] value=" << seatTable[sn_XY[g][0]][sn_XY[g][1]] << "\n";
			//system("pause");
		}

		//有了每個資料點所在的XY座標 就可以進行投票(加減權重)
		for (int x = 0; x < 4; x++)
		{
			for (int y = 0; y < 4; y++)
			{
				//正權重總合 >= 负權重總合 代表负的錯了
				if (seatTable_PN[x][y][0] > seatTable_PN[x][y][1])
				{
					return_Matrix[h] += seatTable_PN[x][y][1];
				}
				//正權重總合 < 负權重總合 代表正的錯了
				else if (seatTable_PN[x][y][0] < seatTable_PN[x][y][1])
				{
					return_Matrix[h] += seatTable_PN[x][y][0];
				}
				//相等時，預設被當成正的，负的被當成ERROR
				else
				{
					return_Matrix[h] += seatTable_PN[x][y][1];;
				}
				
			}

			//cout << "[" << sn_XY[x][0] << "]["<< sn_XY[x][1] << "] value=" << seatTable[sn_XY[x][0]][sn_XY[x][1]] << "\n";
			//system("pause");
		}




		
		//printf("[%d+%d]\n", adaboost_XAxis, adaboost_YAxis);
		//for (int q = 3; q >= 0; q--)
		//{
		//	cout << PorN(seatTable[0][q])
		//		<< "|" << PorN(seatTable[1][q])
		//		<< "|" << PorN(seatTable[2][q])
		//		<< "|" << PorN(seatTable[3][q]) << "\n";
		//}
		//system("pause");

		/*
		for (int j = 0; j < 4; j++)
			cout << seatTable[j][0] << ", " << seatTable[j][1] << ", " << seatTable[j][2] << ", " << seatTable[j][3] << "\n";

		cout << "\nh=" << h << ", returen_Matrix=" << return_Matrix[h];
		system("pause");
		*/


		//找最好的一刀
		//for (int j = 1; j < 10; j++)
		//{
		//	float theta1 = (max - min) / 10 * j;
		//	float error1 = 0;
		//	float polarity1 = 1;
		//	for (int i = 0; i < pf_sn; i++)
		//	{
		//		if (pf[i] < theta1)
		//		{
		//			error1 = error1 + pw[i];
		//		}
		//	}
		//	for (int i = 0; i < nf_sn; i++)
		//	{
		//		if (nf[i] > theta1)
		//		{
		//			error1 = error1 + nw[i];
		//		}
		//	}
		//	if (error1 > 0.5)
		//	{
		//		polarity1 = -1;
		//		error1 = 1 - error1;
		//	}
		//	if (error1 < error)
		//	{
		//		error = error1;
		//		polarity = polarity1;
		//		theta = theta1;
		//	}
		//}
		//float err_WeakLearn[] = { error, polarity, theta };
		//return err_WeakLearn;

		//cout << "pwpwpwpwpwpwp,,,,";
		//for (int i = 0; i < pf_sn; i++)
		//{
		//	cout << pw[i] << ", ";
		//}

		//cout << "\n";
		//system("pause");

		//cout << "nwnwnw,,,,";
		//for (int i = 0; i < nf_sn; i++)
		//{
		//	cout << nw[i] << ", ";
		//}

		//cout << "\n\n";
		//system("pause");

	}



}


