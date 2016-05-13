#include <cstdio>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <ctime>
#include <algorithm>

#include "compute.hpp"
using namespace std;

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
			delete[] bestTable;
		}
};

class ReturnPair
{
	public:
		float previous_R_Sum;   //過去10、5、1分鐘的報酬率總合
		float target_R;			//未來30分鐘的報酬
		int pf_Index;			//正資料索引值
		int nf_Index;			//負資料索引值
};

bool CompareR(ReturnPair, ReturnPair);
void AdaBoostTrain(float[][5000], float[][5000], int [][2], float**);
void AdaBoostTest(float data_Fe[], float data_Re[], float**);

void KNN_Search(float test_Re[], ReturnPair pair_PF[], ReturnPair pair_NF[],
	float arr_Fe_PF[][233590], float arr_Fe_NF[][221629], float arr_Re_PF[][233590], float arr_Re_NF[][221629],
	float real_Fe_PF[][5000], float real_Fe_NF[][5000], float real_Re_PF[][5000], float real_Re_NF[][5000]);

string PorN(float);

const int times = 50;	//訓練次數
const int Train_PF_Num = 233590;  // positive number Traing Data
const int Train_NF_Num = 221629;  // negative number Traing Data
const int KNN_ForTrainData = 5000; //例如:KNN找出一萬筆 正負訓練資料各五千
const int Test_PF_Num = 211;
const int Test_NF_Num = 250;
//const int Test_PF_Num = 18341;	// positive number Testing Data
//const int Test_NF_Num = 18219;  // negative number Testing Data
const int fn = 162;		// feature number
const int rn = 4;		// return number

float success_Count = 0.0;
float fail_Count = 0.0;
float total_Count = 0.0;
float predict_Rate = 0.0;
float total_Profit = 0.0;

fstream file_Output;

//最終的模型(times張表)
Model* model;

//總資料個數total_SampleNumber
int total_sn = KNN_ForTrainData*2;
const int cn2 = fn * (fn - 1) / 2;

int main()
{
	//讀資料是全部都讀 共分兩大類Feautre與Return
	char file_Train_Fe_PF[] = "D:\\Ada_Train\\2001-2012_F_Train_PF.txt";    //233590*162
	char file_Train_Re_PF[] = "D:\\Ada_Train\\2001-2012_Re_Train_PF.txt";   //233590*4
	char file_Train_Fe_NF[] = "D:\\Ada_Train\\2001-2012_F_Train_NF.txt";    //221629*162
	char file_Train_Re_NF[] = "D:\\Ada_Train\\2001-2012_Re_Train_NF.txt";   //221629*4

	//一個禮拜的測試資料 共(165*10)筆分鐘資料
	char file_Test_Fe_PF[] = "D:\\Ada_Test\\2013_F_Test_PF.txt";	   //418*162
	char file_Test_Re_PF[] = "D:\\Ada_Test\\2013_Re_Test_PF.txt";	   //418*4
	char file_Test_Fe_NF[] = "D:\\Ada_Test\\2013_F_Test_NF.txt";	   //349*162
	char file_Test_Re_NF[] = "D:\\Ada_Test\\2013_Re_Test_NF.txt";	   //349*4 

	//char file_Test_Re_PF[] = "G:\\2013_Re_Test_PF.txt";	   //18341*4
	//char file_Test_Fe_PF[] = "G:\\2013_F_Test_PF.txt";	   //18341*162
	//char file_Test_Re_NF[] = "G:\\2013_Re_Test_NF.txt";	   //18219*4 
	//char file_Test_Fe_NF[] = "G:\\2013_F_Test_NF.txt";	   //18219*162

	//輸出檔案
	char file_Result[] = "JointAdaboost_GPU_30分鐘_5000knn_訓練3次.csv";
	file_Output.open(file_Result, ios::out);//開啟檔案
	if (!file_Output){//如果開啟檔案失敗，file_Output為0；成功，file_Output為非0
		cout << "Fail to open file: " << file_Result << endl;
	}

	//最原始的Traing Data 數量龐大
	auto arr_Train_Fe_PF = new float[fn][Train_PF_Num];
	auto arr_Train_Fe_NF = new float[fn][Train_NF_Num];
	auto arr_Train_Re_PF = new float[rn][Train_PF_Num];
	auto arr_Train_Re_NF = new float[rn][Train_NF_Num];

	//真正去學習的只有從KNN找出的一萬筆 正負各給五千
	auto real_Train_Fe_PF = new float[fn][KNN_ForTrainData];
	auto real_Train_Fe_NF = new float[fn][KNN_ForTrainData];
	auto real_Train_Re_PF = new float[rn][KNN_ForTrainData];
	auto real_Train_Re_NF = new float[rn][KNN_ForTrainData];

	auto arr_Test_Fe_PF = new float[fn][Test_PF_Num];
	auto arr_Test_Fe_NF = new float[fn][Test_NF_Num];
	auto arr_Test_Re_PF = new float[rn][Test_PF_Num];
	auto arr_Test_Re_NF = new float[rn][Test_NF_Num];

	//KNN Search Function中所需要的資料結構 
	//將(1)過去報酬率 (2)未來報酬率 (3)索引值 綁定後再做排序
	auto pair_PF = new ReturnPair[Train_PF_Num];
	auto pair_NF = new ReturnPair[Train_NF_Num];

	/*
	利用'\t'當作分隔符號
	將[rows][feature]轉置成[feature][rows]方便平行運算的Code處理
	有Feature以及Return兩種資料需要處理
	*/

	fstream fp1;
	char line1[256];

	fp1.open(file_Train_Fe_PF, ios::in);//開啟檔案
	if (!fp1){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Train_Fe_PF << endl;
	}

	int i1 = 0;
	int j1 = 0;

	while (fp1.getline(line1, sizeof(line1), '\t'))
	{
		if (i1 == Train_PF_Num)
			break;

		//atof是一種將字串轉為浮點數的函數
		arr_Train_Fe_PF[j1][i1] = atof(line1);

		j1++;

		if (j1 == fn)
		{
			j1 = 0;
			i1++;
		}
	}
	fp1.close();//關閉檔案

	fp1.open(file_Train_Re_PF, ios::in);//開啟檔案
	if (!fp1){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Train_Re_PF << endl;
	}

	i1 = 0;
	j1 = 0;

	while (fp1.getline(line1, sizeof(line1), '\t'))
	{
		if (i1 == Train_PF_Num)
			break;

		//atof是一種將字串轉為浮點數的函數
		arr_Train_Re_PF[j1][i1] = atof(line1);

		j1++;

		if (j1 == rn)
		{
			j1 = 0;
			i1++;
		}
	}
	fp1.close();//關閉檔案



	fstream fp2;
	char line2[256];

	fp2.open(file_Train_Fe_NF, ios::in);//開啟檔案
	if (!fp2){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Train_Fe_NF << endl;
	}

	int i2 = 0;
	int j2 = 0;

	while (fp2.getline(line2, sizeof(line2), '\t'))
	{
		if (i2 == Train_NF_Num)
			break;

		arr_Train_Fe_NF[j2][i2] = atof(line2);

		j2++;

		if (j2 == fn)
		{
			j2 = 0;
			i2++;
		}
	}

	fp2.close();//關閉檔案

	fp2.open(file_Train_Re_NF, ios::in);//開啟檔案
	if (!fp2){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Train_Re_NF << endl;
	}

	i2 = 0;
	j2 = 0;

	while (fp2.getline(line2, sizeof(line2), '\t'))
	{
		if (i2 == Train_NF_Num)
			break;

		arr_Train_Re_NF[j2][i2] = atof(line2);

		j2++;

		if (j2 == rn)
		{
			j2 = 0;
			i2++;
		}
	}

	fp2.close();//關閉檔案


	fstream fp3;
	char line3[256];

	fp3.open(file_Test_Fe_PF, ios::in);//開啟檔案
	if (!fp3){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Test_Fe_PF << endl;
	}

	int i3 = 0;
	int j3 = 0;

	while (fp3.getline(line3, sizeof(line3), '\t'))
	{
		if (i3 == Test_PF_Num)
			break;

		arr_Test_Fe_PF[j3][i3] = atof(line3);

		j3++;

		if (j3 == fn)
		{
			j3 = 0;
			i3++;
		}
	}

	fp3.close();//關閉檔案

	fp3.open(file_Test_Re_PF, ios::in);//開啟檔案
	if (!fp3){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Test_Re_PF << endl;
	}

	i3 = 0;
	j3 = 0;

	while (fp3.getline(line3, sizeof(line3), '\t'))
	{
		if (i3 == Test_PF_Num)
			break;

		arr_Test_Re_PF[j3][i3] = atof(line3);

		j3++;

		if (j3 == rn)
		{
			j3 = 0;
			i3++;
		}
	}

	fp3.close();//關閉檔案

	//for (size_t i = 0; i < Test_PF_Num; i++)
	//{
	//	cout << arr_Test_Re_PF[0][i] << "\n";
	//	cout << arr_Test_Re_PF[1][i] << "\n";
	//	cout << arr_Test_Re_PF[2][i] << "\n";
	//	cout << arr_Test_Re_PF[3][i] << "\n";
	//	system("pause");
	//}

	fstream fp4;
	char line4[256];

	fp4.open(file_Test_Fe_NF, ios::in);//開啟檔案
	if (!fp4){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Test_Fe_NF << endl;
	}

	int i4 = 0;
	int j4 = 0;

	while (fp4.getline(line4, sizeof(line4), '\t'))
	{
		if (i4 == Test_NF_Num)
			break;

		arr_Test_Fe_NF[j4][i4] = atof(line4);

		j4++;

		if (j4 == fn)
		{
			j4 = 0;
			i4++;
		}
	}

	fp4.close();//關閉檔案

	fp4.open(file_Test_Re_NF, ios::in);//開啟檔案
	if (!fp4){//如果開啟檔案失敗，fp為0；成功，fp為非0
		cout << "Fail to open file: " << file_Test_Re_NF << endl;
	}

	i4 = 0;
	j4 = 0;

	while (fp4.getline(line4, sizeof(line4), '\t'))
	{
		if (i4 == Test_NF_Num)
			break;

		arr_Test_Re_NF[j4][i4] = atof(line4);

		j4++;

		if (j4 == rn)
		{
			j4 = 0;
			i4++;
		}
	}

	fp4.close();//關閉檔案

	//for (size_t i = 0; i < Test_NF_Num; i++)
	//{
	//	cout << arr_Test_Re_NF[0][i] << "\n";
	//	cout << arr_Test_Re_NF[1][i] << "\n";
	//	cout << arr_Test_Re_NF[2][i] << "\n";
	//	cout << arr_Test_Re_NF[3][i] << "\n";
	//	system("pause");
	//}
	
	//訓練幾次就有幾個Model
	model = new Model[times];

	//CN取2的清單
	int list[cn2][2];

	/*
		list[][] C3取2 示意圖
		---------------------
		[0][1]
		[0][2]
		[1][2]
	*/

	//處理CN取2的實作 結果存於list
	int start = 0;
	int start_Nei = start + 1;
	for (int i = 0; i < cn2; ++i)
	{
		list[i][0] = start;
		list[i][1] = start_Nei;

		if (start_Nei + 1 >= fn)
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


	float **q_Map;
	//四分位數的Map 二維陣列
	q_Map = new float*[fn];
	for (int x = 0; x < fn; x++)
		q_Map[x] = new float[3];

	//看有Testing Data有幾筆分鐘資料 就訓練幾次Model
	for (size_t i = 0; i < Test_PF_Num + Test_NF_Num; i++)
	{
		auto return_Test = new float[rn];
		auto feature_Test = new float[fn];

		//postive
		if (i < Test_PF_Num)
		{
			for (size_t a = 0; a < rn; a++)
			{
				return_Test[a] = arr_Test_Re_PF[a][i];
			}
			for (size_t b = 0; b < fn; b++)
			{
				feature_Test[b] = arr_Test_Fe_PF[b][i];
			}
		}
		//negative
		else if (i < Test_PF_Num + Test_NF_Num)
		{
			for (size_t a = 0; a < rn; a++)
			{
				return_Test[a] = arr_Test_Re_NF[a][i - Test_PF_Num];
			}
			for (size_t b = 0; b < fn; b++)
			{
				feature_Test[b] = arr_Test_Fe_NF[b][i - Test_PF_Num];
			}
		}

		KNN_Search(return_Test, pair_PF, pair_NF,
			arr_Train_Fe_PF, arr_Train_Fe_NF, arr_Train_Re_PF, arr_Train_Re_NF,
			real_Train_Fe_PF, real_Train_Fe_NF, real_Train_Re_PF, real_Train_Re_NF);



		/*
		q_Map示意圖
		-----------
		Q1	Q2	Q3
		1   2	5	7
		2	8	12	16
		3
		.
		.
		.
		fn
		*/

		//正副資料結合成一大條暫時的陣列
		float *temp_Range = new float[total_sn];

		//算四分位
		for (int g = 0; g < fn; g++)
		{
			//讀正資料進來
			for (int z = 0; z < KNN_ForTrainData; z++)
				temp_Range[z] = real_Train_Fe_PF[g][z];

			//讀负資料進來
			for (int z = 0; z < KNN_ForTrainData; z++)
				temp_Range[z + KNN_ForTrainData] = real_Train_Fe_NF[g][z];

			/*
			cout << "檢查還沒排序過的temp_Range";
			for (int z = 0; z < total_sn; z++)
			cout << "" << temp_Range[z] << ", ";
			cout << "\n";
			*/

			//利用內建Library排序瞜 傳入頭尾即可
			sort(temp_Range, temp_Range + total_sn);

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
				q_Map[g][0] = temp_Range[(int)q1_Index - 1];
				q_Map[g][2] = temp_Range[(int)q3_Index - 1];
			}
			else
			{
				q_Map[g][0] = (temp_Range[(int)q1_Index - 1] + temp_Range[(int)q1_Index]) / 2;
				q_Map[g][2] = (temp_Range[(int)q3_Index - 1] + temp_Range[(int)q3_Index]) / 2;
			}

			//Q2除以四後還要乘二，獨立處理
			if ((total_sn + 1) % 2 == 0)
			{
				q_Map[g][1] = temp_Range[(int)q2_Index - 1];
			}
			else
			{
				q_Map[g][1] = (temp_Range[(int)q2_Index - 1] + temp_Range[(int)q2_Index]) / 2;
			}

			//cout << "檢查qIndex\n";
			//cout << (int)q1_Index << ", " << (int)q2_Index << ", " << (int)q3_Index << "\n";

			//將資料數值轉成座標
			for (size_t k = 0; k < KNN_ForTrainData; k++)
			{
				if (real_Train_Fe_PF[g][k] < q_Map[g][0])
					real_Train_Fe_PF[g][k] = 0;
				else if (real_Train_Fe_PF[g][k] < q_Map[g][1])
					real_Train_Fe_PF[g][k] = 1;
				else if (real_Train_Fe_PF[g][k] < q_Map[g][2])
					real_Train_Fe_PF[g][k] = 2;
				else
					real_Train_Fe_PF[g][k] = 3;

				if (real_Train_Fe_NF[g][k] < q_Map[g][0])
					real_Train_Fe_NF[g][k] = 0;
				else if (real_Train_Fe_NF[g][k] < q_Map[g][1])
					real_Train_Fe_NF[g][k] = 1;
				else if (real_Train_Fe_NF[g][k] < q_Map[g][2])
					real_Train_Fe_NF[g][k] = 2;
				else
					real_Train_Fe_NF[g][k] = 3;
			}
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

		clock_t train_beginTime = clock();
		//參數說明: 正資料pointer, 负資料pointer, 正資料sample數, 负資料sample數, 特徵數量, 訓練次數, CN取2的清單pointer
		AdaBoostTrain(real_Train_Fe_PF, real_Train_Fe_NF, list, q_Map);
		clock_t train_endTime = clock();
		double train_Sec = double(train_endTime - train_beginTime) / CLOCKS_PER_SEC;
		cout << "Adaboost Train: " << train_Sec << " seconds!!!!" << endl;

		AdaBoostTest(feature_Test, return_Test, q_Map);

		//釋放記憶體
		delete[] return_Test;
		delete[] feature_Test;
	

	}


	for (int x = 0; x < fn; x++)
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
void AdaBoostTrain(float pf[][5000], float nf[][5000], int list[][2], float** q_Map)
{
	float *pw = new float[KNN_ForTrainData];	//正資料權重
	float *nw = new float[KNN_ForTrainData];	//负資料權重
	float wsum = 0;

	//初使化正负臉權重 各資料權重一開始相同
	for (int i = 0; i < KNN_ForTrainData; i++)
	{
		pw[i] = 1.0 / (total_sn);
		nw[i] = 1.0 / (total_sn);
	}
	
	for (int i = 0; i < KNN_ForTrainData; i++)
	{
		wsum += pw[i];
		wsum += nw[i];
	}

	for (int i = 0; i < KNN_ForTrainData; i++)
	{
		pw[i] /= wsum;
		nw[i] /= wsum;
	}

	/*
		cout << "檢查 pw 正資料權重";
		for (int i = 0; i < KNN_ForTrainData; i++)
			cout << pw[i] << ", ";

		cout << "\n";
		system("pause");

		cout << "檢查 nw 负資料權重";
		for (int i = 0; i < KNN_ForTrainData; i++)
			cout << nw[i] << ", ";

		cout << "總權重 wsum \n";
		cout << "wsum: " << wsum;

		system("pause");
	*/

	//OpenCL, 多傳一個List進去吧

	Compute compute("JointLearn", CL_DEVICE_TYPE_GPU);
	int pf_shape[2] = { fn, KNN_ForTrainData };
	int nf_shape[2] = { fn, KNN_ForTrainData };
	float ret[cn2];

	compute.set_buffer((float *)pf, fn * KNN_ForTrainData*sizeof(float));
	compute.set_buffer((float *)nf, fn * KNN_ForTrainData*sizeof(float));

	compute.set_buffer(pw, KNN_ForTrainData * sizeof(float));
	compute.set_buffer(nw, KNN_ForTrainData * sizeof(float));
	
	compute.set_buffer(fn);
	compute.set_buffer(KNN_ForTrainData);
	compute.set_buffer(KNN_ForTrainData);

	compute.set_buffer((float *)list, cn2 * 2 * sizeof(int));
	compute.set_buffer((float *)q_Map, fn * 3 * sizeof(float));

	compute.set_ret_buffer((float *)ret, cn2 * sizeof(float));

	//做times次的WeakLearner，換句話說，在此之上的程式碼只會執行一次
	for (int i = 0; i < times; i++)
	{
		wsum = 0;

		//調整權重
		for (int i = 0; i < KNN_ForTrainData; i++)
		{
			wsum += pw[i];
			wsum += nw[i];
		}
			
		for (int i = 0; i < KNN_ForTrainData; i++)
		{
			pw[i] /= wsum;
			nw[i] /= wsum;
		}
					
		/*
			cout << "檢查 pw 正資料權重";
			for (int i = 0; i < KNN_ForTrainData; i++)
				cout << pw[i] << ", ";

			cout << "\n";
			system("pause");

			cout << "檢查 nw 负資料權重";
			for (int i = 0; i < KNN_ForTrainData; i++)
				cout << nw[i] << ", ";

			cout << "總權重 wsum \n";
			cout << "wsum: " << wsum;
			system("pause");
		*/

		//輸出的陣列 存WeakLearn跑完輸出的各Feature錯誤率 
		//float *err_WeakLearn = new float[cn2];
		
		//每一個資料點的座標 一對一的記錄 所以有正臉數+副臉數
		//sn_XY會傳入WeakLearn進行計算
		//int **sn_XY = new int*[total_sn];
		//for (int k = 0; k < total_sn; k++)
		//	sn_XY[k] = new int[2];

		//WeakLearn(pf, nf,pw, nw, list, err_WeakLearn, sn_XY);

		compute.reset_buffer(2, pw);
		compute.reset_buffer(3, nw);
		compute.run(cn2);

		float error = 1;
		int selectif = -1;
		float beta;

		for (int k = 0; k < cn2; k++)
		{
			if (ret[k] < error)
			{
				error = ret[k];
				selectif = k;
			}
		}

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
			if (g < KNN_ForTrainData)
				bestTable[(int)pf[indX][g]][(int)pf[indY][g]] += pw[g];
			else
				bestTable[(int)nf[indX][g]][(int)nf[indY][g]] -= nw[g - KNN_ForTrainData];
		}

		//正臉調權重
		for (int i = 0; i < KNN_ForTrainData; i++)
		{			
			if (bestTable[(int)pf[indX][i]][(int)pf[indY][i]] >= 0)
			{
				pw[i] = pw[i] * beta;
			}
		}

		//負臉調權重
		for (int i = KNN_ForTrainData; i < total_sn; i++)
		{
			if (bestTable[(int)nf[indX][i]][(int)nf[indY][i]] < 0)
			{
				nw[i - KNN_ForTrainData] = nw[i - KNN_ForTrainData] * beta;
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

		model[i].alpha = log(1.0 / beta);

		//cout << "beta: " << beta << "\n";
		//cout << "alpha" << log(1.0 / beta) << "\n";

		//if (i % 50 == 0)
		//printf("[%d+%d] , %d, %f\n", list[selectif][0], list[selectif][1], selectif + 1, model[i].alpha);
				
		//for (int q = 0; q < 4; q++)
		//{
		//	cout << PorN(bestTable[q][0])
		//		<< "|" << PorN(bestTable[q][1])
		//		<< "|" << PorN(bestTable[q][2])
		//		<< "|" << PorN(bestTable[q][3]) << "\n";
		//}

		//釋放記憶體
		//for (int i = 0; i < total_sn; i++)
		//	delete[] sn_XY[i];

		//delete[] sn_XY;

		//compute.reset_buffer(2, pw);
		//compute.reset_buffer(3, nw);
	}

	delete[] pw;
	delete[] nw;
}

string PorN(float num)
{
	if (num >= 0)
		return "+";
	else
		return "-";
}


void AdaBoostTest(float data_Fe[], float data_Re[], float** q_Map)
{
	float alphaSum = 0;
	float score = 0;
	
	for (size_t i = 0; i < times; i++)
	{
		alphaSum += model[i].alpha;

		int x = model[i].x;
		int y = model[i].y;

		//sn_XY會傳入WeakLearn進行計算
		int test_XY[2];

		if (data_Fe[x] < q_Map[x][0])
			test_XY[0] = 0;
		else if (data_Fe[x] < q_Map[x][1])
			test_XY[0] = 1;
		else if (data_Fe[x] < q_Map[x][2])
			test_XY[0] = 2;
		else
			test_XY[0] = 3;

		if (data_Fe[y] < q_Map[y][0])
			test_XY[1] = 0;
		else if (data_Fe[y] < q_Map[y][1])
			test_XY[1] = 1;
		else if (data_Fe[y] < q_Map[y][2])
			test_XY[1] = 2;
		else
			test_XY[1] = 3;

		//cout << "BestTable:" << i << " " <<model[i].bestTable[test_XY[0]][test_XY[1]] << "\n";

		if (model[i].bestTable[test_XY[0]][test_XY[1]] >= 0)
			score += model[i].alpha;
	}

	float finalDecision = score / alphaSum;
	//cout << "score= " << score << " alphaSum= " << alphaSum << " Final Decision" << finalDecision <<"\n";

	//看多
	if (finalDecision >= 0.5)
	{
		if (data_Re[3] >= 0)
		{
			success_Count++;
			total_Profit += abs(data_Re[3]);
		}
		else
		{
			total_Profit -= abs(data_Re[3]);
			fail_Count++;
		}

		file_Output << finalDecision << "," << data_Re[3] << ",\n";
	}
	//看空
	else
	{
		if (data_Re[3] < 0)
		{
			success_Count++;
			total_Profit += abs(data_Re[3]);
		}
		else
		{
			total_Profit -= abs(data_Re[3]);
			fail_Count++;
		}

		file_Output << finalDecision << "," << -data_Re[3] << ",\n";
	}

	total_Count++;

	//cout << "\nsocre= " << score;
	//cout << "\nalphaSum = " << alphaSum;
	cout << "\nfinalDecision= " << finalDecision;
	cout << "\n總獲利= " << total_Profit;
	cout << "\n勝率= " << success_Count / total_Count;

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

//根據原始資料 尋找相似的10,000筆資料出來建模 (正負資料各五千)
//傳入的資料結構 列=Feature, 欄=分鐘資料
void KNN_Search(float test_Re[], ReturnPair pair_PF[], ReturnPair pair_NF[],
	float arr_Fe_PF[][233590], float arr_Fe_NF[][221629], float arr_Re_PF[][233590], float arr_Re_NF[][221629],
	float real_Fe_PF[][5000], float real_Fe_NF[][5000], float real_Re_PF[][5000], float real_Re_NF[][5000])
{
	// p means previous, t means target, 測試的這一分鐘之四個Return宣告如下
	float return_p10 = test_Re[0];
	float return_p5 = test_Re[1];
	float return_p1 = test_Re[2];
	float return_t30 = test_Re[3];

	////cout << "\n" << return_p10 << ", " << return_p5 << ", " << return_p1 << ", " << return_t30 << "\n";
	////system("pause");

	int zero = 0;

	//從所有歷史資料中，根據現在Testing這筆資料之報酬率，計算距離總合與我最近的(最像的)
	for (size_t i = 0; i < Train_PF_Num; i++)
	{
		pair_PF[i].previous_R_Sum = abs(return_p10 - arr_Re_PF[0][i])
			+ abs(return_p5 - arr_Re_PF[1][i])
			+ abs(return_p1 - arr_Re_PF[2][i]);
		pair_PF[i].target_R = arr_Re_PF[3][i];
		pair_PF[i].pf_Index = i;

		//cout << pair_PF[i].previous_R_Sum << "\n";
		//cout << pair_PF[i].target_R << "\n";
		//system("pause");
	}

	//從所有歷史資料中，根據現在Testing這筆資料之報酬率，計算距離總合與我最近的(最像的)
	for (size_t j = 0; j < Train_NF_Num; j++)
	{
		pair_NF[j].previous_R_Sum = abs(return_p10 - arr_Re_NF[0][j])
			+ abs(return_p5 - arr_Re_NF[1][j])
			+ abs(return_p1 - arr_Re_NF[2][j]);

		pair_NF[j].target_R = arr_Re_NF[3][j];
		pair_NF[j].nf_Index = j;
	}

	//檢查排序前的內容內容
	//cout << "UnSorted: " << "\n";
	//for (size_t i = 0; i < 30000; i++)
	//	cout << pair_PF[i].previous_R_Sum << "\n";

	//自定義排序方法
	sort(pair_PF, pair_PF + Train_PF_Num, CompareR);
	sort(pair_NF, pair_NF + Train_NF_Num, CompareR);

	//排序後 就可以抓出正負最像的那五千筆
	for (size_t i = 0; i < KNN_ForTrainData; i++)
	{
		int pf_Idx = pair_PF[i].pf_Index;
		int nf_Idx = pair_NF[i].nf_Index;

		//cout << "Feature PF NF \n";

		//找出相對應的特徵
		for (size_t f = 0; f < fn; f++)
		{
			real_Fe_PF[f][i] = arr_Fe_PF[f][pf_Idx];
			real_Fe_NF[f][i] = arr_Fe_NF[f][nf_Idx];

			//cout << real_Fe_PF[f][i] << "\n";
			//cout << real_Fe_NF[f][i] << "\n";
		}

		//cout << "Return PF NF \n";

		//找出相對應的報酬率
		for (size_t r = 0; r < rn; r++)
		{
			real_Re_PF[r][i] = arr_Re_PF[r][pf_Idx];
			real_Re_NF[r][i] = arr_Re_NF[r][nf_Idx];

			//cout << real_Re_PF[r][i] << "\n";
			//cout << real_Re_NF[r][i] << "\n";
		}

		//cout << "PF\n";
		//cout << "Distance Sum " << pair_PF[i].previous_R_Sum << "\n";
		//cout << "Target " << pair_PF[i].target_R << "\n";
		//cout << "Index " << pair_PF[i].pf_Index << "\n";

		//cout << "NF\n";
		//cout << "Distance Sum " << pair_NF[i].previous_R_Sum << "\n";
		//cout << "Target " << pair_NF[i].target_R << "\n";
		//cout << "Index " << pair_NF[i].nf_Index << "\n";

		//if (i % 100 == 0)
		//	system("pause");

		//if (pair_PF[i].previous_R_Sum == 0)
		//{
		//	cout << " =0 " << "\n";
		//}
		//else if (pair_PF[i].previous_R_Sum < 0)
		//{
		//	cout << " <0 " << "\n";
		//}
		//else
		//{
		//	cout << " >0 " << "\n";
		//}

		//printf("%f", pair_PF[i].previous_R_Sum);
		//system("pause");
	}
	//system("pause");
}

bool CompareR(ReturnPair r1, ReturnPair r2)
{
	return r1.previous_R_Sum < r2.previous_R_Sum;
}


