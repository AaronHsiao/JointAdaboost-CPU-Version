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


void WeakLearn(float[][15], float[][14], float[], float[], int, int, int, int[][2], float*, float**);
void AdaBoostTrain(float pf[][15], float nf[][14], int pf_sn, int nf_sn, int fn, int times, int list[][2]);
int AdaBoostTest(float data[], int data_sn, int data_fn);
float MyRound(float);
string PorN(float);

Compute compute("WeakLearn", CL_DEVICE_TYPE_GPU);

const int times = 3000;	//訓練次數
float F[times][4];	//用二維矩陣 存放每次訓練完之結果 4分別代表著 1. selectif(選到的Feature) 2. polarity(右邊是正or負資料) 3. error(錯誤率) 4. alpha值 

int main()
{
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

	const int feature_Size = 3;		//特徵數量
	const int pf_sample_Size = 15;	//正臉Sample數量
	const int nf_sample_Size = 14;	//負臉Sample數量

	//PF Sample Data
	float TrainPF[feature_Size][pf_sample_Size] = { {1, 2, 7, 7, 7, 4, 5, 8, 6, 5, 9, 10, 3, 6, 8},
													{7, 9, 3, 4, 6, 2, 2, 0, 1, 3, 4, 5, 7, 10, 2 },
													{3, 6, 6, 6, 2, 4, 3, 1, 5, 8, 9, 5, 7, 9, 3 } };

	//NF Sample Data
	float TrainNF[feature_Size][nf_sample_Size] = { { 9, 1, 3, 4, 6, 2, 2, 2, 2, 7, 0, 8, 1, 3},
													{ 7, 7, 4, 6, 3, 5, 6, 3, 2, 8, 9, 3, 1, 2},
													{ 8, 5, 0, 10, 10, 2, 4, 1, 3, 5, 7, 2, 0, 9} };

	//訓練次數為CN取2次
	const int times = feature_Size*(feature_Size - 1) / 2;

	//list[][] C3取2 示意圖
	/*
		[0][1]
		[0][2]
		[1][2]
	*/

	//CN取2的清單
	int list[times][2];

	//處理CN取2的實作 結果存於list
	int start = 0;
	int start_Nei = start + 1;
	for (int i = 0; i < times;++i)
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
	AdaBoostTrain(TrainPF, TrainNF, pf_sample_Size, nf_sample_Size, feature_Size, times, list);

	//int TP = AdaBoostTest(arr_Test_PF1[0], row_Test_PF1, column_Test_NF1);
	//printf("%f/n", TP / row_Test_PF1);
	//int FP = row_Test_NF1-AdaBoostTest(arr_Test_NF1[0], row_Test_NF1, column_Test_NF1);
	//printf("%f/n", FP / row_Test_NF1);
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Time: " << elapsed_secs << " seconds!!!!" << endl;

	////釋放記憶體
	//for (int i = 0; i < row_Train_PF1; i++)
	//{
	//	delete[] arr_Train_PF1[i];
	//}

	//delete[] arr_Train_PF1;
	//delete[] arr_Train_NF1;

	system("pause");
}

//參數說明: 正資料pointer, 负資料pointer, 正資料sample數, 负資料sample數, 特徵數量, 訓練次數, CN取2的清單pointer
void AdaBoostTrain(float pf[][15], float nf[][14], int pf_sn, int nf_sn, int fn, int times, int list[][2])
{
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
	float **q_Map = new float*[fn];

	for (int x = 0; x < fn;	x++)
		q_Map[x] = new float[3];


	//總資料個數total_SampleNumber
	int total_sn = pf_sn + nf_sn;
	
	//算四分位
	for (int g = 0; g < fn; g++)
	{
		//正副資料結合成一大條暫時的陣列
		float *temp_Range = new float[total_sn];

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


	//做times次的WeakLeaner，換句話說，在此之上的程式碼只會執行一次
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
		
		//compute.reset_buffer(2, pw);
		//compute.reset_buffer(3, nw);

		//幾個Kernel在跑
		//compute.run(2101);
		
		/*
			參數說明 正資料pointer, 负資料pointer, 正資料權重, 负資料權重, 正資料個數, 负資料個數,
			特徵數量, 組合式特徵清單, 回傳陣列, q_Map pointer
		*/
		WeakLearn(pf, nf,pw, nw, pf_sn, nf_sn, fn, list, err_WeakLearn, q_Map);

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
		}

		cout << pw[0] * 4 << "\n";
		cout << "err_WeakLearn: " <<err_WeakLearn[0];
		system("pause");

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

		//計票用的4*4的16宮格
		float seatTable[4][4] = { 0 };

		//每一個資料點的座標 一對一的記錄 所以有正臉數+副臉數
		int **sn_XY = new int*[total_sn];
		for (int k = 0; k < total_sn; k++)
			sn_XY[k] = new int[2];

		int indX = list[selectif][0];
		int indY = list[selectif][1];

		//拿出正臉資料 取得正臉座標
		for (int z = 0; z < pf_sn; z++)
		{
			if (pf[indX][z] < q_Map[indX][0])
				sn_XY[z][0] = 0;
			else if (pf[indX][z] < q_Map[indX][1])
				sn_XY[z][0] = 1;
			else if (pf[indX][z] < q_Map[indX][2])
				sn_XY[z][0] = 2;
			else
				sn_XY[z][0] = 3;

			if (pf[indY][z] < q_Map[indY][0])
				sn_XY[z][1] = 0;
			else if (pf[indY][z] < q_Map[indY][1])
				sn_XY[z][1] = 1;
			else if (pf[indY][z] < q_Map[indY][2])
				sn_XY[z][1] = 2;
			else
				sn_XY[z][1] = 3;
		}

		//拿出負臉資料 取得副臉座標
		for (int z = pf_sn; z < nf_sn + pf_sn; z++)
		{
			if (nf[indX][z - pf_sn] < q_Map[indX][0])
				sn_XY[z][0] = 0;
			else if (nf[indX][z - pf_sn] < q_Map[indX][1])
				sn_XY[z][0] = 1;
			else if (nf[indX][z - pf_sn] < q_Map[indX][2])
				sn_XY[z][0] = 2;
			else
				sn_XY[z][0] = 3;

			if (nf[indY][z - pf_sn] < q_Map[indY][0])
				sn_XY[z][1] = 0;
			else if (nf[indY][z - pf_sn] < q_Map[indY][1])
				sn_XY[z][1] = 1;
			else if (nf[indY][z - pf_sn] < q_Map[indY][2])
				sn_XY[z][1] = 2;
			else
				sn_XY[z][1] = 3;
		}

		//投票瞜
		for (int g = 0; g < total_sn; g++)
		{
			if (g < pf_sn)
				seatTable[sn_XY[g][0]][sn_XY[g][1]] += pw[g];
			else
				seatTable[sn_XY[g][0]][sn_XY[g][1]] -= nw[g - pf_sn];
		}

		//正臉調權重
		for (int i = 0; i < pf_sn; i++)
		{			
			if (seatTable[sn_XY[i][0]][sn_XY[i][1]] >= 0)
			{
				pw[i] = pw[i] * beta;
			}
		}

		//負臉調權重
		for (int i = pf_sn; i < nf_sn+pf_sn; i++)
		{
			if (seatTable[sn_XY[i][0]][sn_XY[i][1]] < 0)
			{
				nw[i-pf_sn] = nw[i-pf_sn] * beta;
			}
		}


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

		/*if (i % 100 == 0)*/
		printf("[%d+%d] , %f, %f\n", list[i][0], list[i][1], selectif + 1, log(1 / beta));
		for (int q = 0; q < 4; q++)
		{
			//cout << PorN(seatTable[q][0])
			//	<< "|" << PorN(seatTable[q][1])
			//	<< "|" << PorN(seatTable[q][2])
			//	<< "|" << PorN(seatTable[q][3]) << "\n";
		}
		

	}

	for (int x = 0; x < fn;	x++)
	{
		delete[] q_Map[x];
	}

	delete[] q_Map;

	//釋放記憶體
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

int AdaBoostTest(float data[], int data_sn, int data_fn)
{
	float *predit = new float[data_sn];
	int count = 0;
	for (int i = 0; i < data_sn; i++)
	{
		for (int j = 0; j < data_fn; j++)
		{
			for (int k = 0; k < sizeof(F) / sizeof(F[0]); k++)
			{
				if (F[k][1] == 1)
				{
					if (data[i*(data_sn - 1) + j] >= F[k][2])
					{
						predit[i] = predit[i] + F[k][3];
					}
				}
				else
				{
					if (data[i*(data_sn - 1) + j] < F[k][2])
					{
						predit[i] = predit[i] + F[k][3];
					}
				}
			}
		}
		if (predit[i] > 0.5)
		{
			count++;
		}
	}
	return count;
}

/*
參數說明 正資料pointer, 负資料pointer, 正資料權重, 负資料權重,
			正資料個數, 负資料個數, 特徵數量, 組合式特徵清單, 回傳陣列, q_Map pointer 
*/
void WeakLearn(float pf[][15], float nf[][14], float pw[], float nw[], 
	int pf_sn, int nf_sn, int fn, int list[][2], float* return_Matrix, float** q_Map)
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

		int **sn_XY = new int*[total_sn];
		for (int k = 0; k < total_sn; k++)
			sn_XY[k] = new int[2];

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
				seatTable[sn_XY[g][0]][sn_XY[g][1]] += pw[g];
				return_Matrix[h] += pw[g];
			}
				
			else
			{
				seatTable[sn_XY[g][0]][sn_XY[g][1]] -= nw[g - pf_sn];
				return_Matrix[h] += nw[g - pf_sn];
			}

			//cout << "[" << sn_XY[g][0] << "]["<< sn_XY[g][1] << "] value=" << seatTable[sn_XY[g][0]][sn_XY[g][1]] << "\n";
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

		//for (int j = 0; j < 4; j++)
		//	cout << seatTable[j][0] << ", " << seatTable[j][1] << ", " << seatTable[j][2] << ", " << seatTable[j][3] << "\n";

		//system("pause");
		


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

		for (int i = 0; i < total_sn; i++)
			delete[] sn_XY[i];

		delete[] sn_XY;

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


