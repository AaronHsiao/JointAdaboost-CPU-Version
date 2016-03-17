#include <cstdio>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <ctime>
#include <algorithm>

using namespace std;

#include "compute.hpp"

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
void AdaBoostTrain(float pf[][15], float nf[][14], int pf1_sn, int nf1_sn, int fn, int times, int list[][2]);
int AdaBoostTest(float data[], int data_sn, int data_fn);
float MyRound(float);
string PorN(float);

Compute compute("WeakLearn", CL_DEVICE_TYPE_GPU);

const int times = 3000;	//訓練次數
float F[times][4];	//用二維矩陣 存放每次訓練完之結果 4分別代表著 1. selectif(選到的Feature) 2. polarity(右邊是正or負資料) 3. error(錯誤率) 4. alpha值 

int main()
{
	////float *output = weakLearn(pf[0], nf[0], pw[0], nw[0], sizeof(pf) / sizeof(pf[0]), sizeof(nf) / sizeof(nf[0]));

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








	const int feature_Size = 3;
	const int pf_sample_Size = 15;

	//PF Sample Data
	float TrainPF[feature_Size][pf_sample_Size] = { {1, 2, 7, 7, 7, 4, 5, 8, 6, 5, 9, 10, 3, 6, 8},
													{7, 9, 3, 4, 6, 2, 2, 0, 1, 3, 4, 5, 7, 10, 2 },
													{3, 6, 6, 6, 2, 4, 3, 1, 5, 8, 9, 5, 7, 9, 3 } };
	//模擬C3取2
	int i = 1;

	const int nf_sample_Size = 14;

	//NF Sample Data
	float TrainNF[feature_Size][nf_sample_Size] = { { 9, 1, 3, 4, 6, 2, 2, 2, 2, 7, 0, 8, 1, 3},
													{ 7, 7, 4, 6, 3, 5, 6, 3, 2, 8, 9, 3, 1, 2},
													{ 8, 5, 0, 10, 10, 2, 4, 1, 3, 5, 7, 2, 0, 9} };

	//Pair Table
	/*
	[1 + 2 ] -> (5, 1), (10, 2), ... (5, 5)
	[1+3] -> ()
	*/

	/*
		改傳C幾取幾的List
		整數的二維矩陣
	*/

	const int times = feature_Size*(feature_Size - 1) / 2;

	int list[times][2];


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

	for (int j = 0; j < times; ++j)
	{
		cout << list[j][0] << ", " << list[j][1] << endl;
	}
	
	


	clock_t begin = clock();
	
	


	AdaBoostTrain(TrainPF, TrainNF, pf_sample_Size, nf_sample_Size, feature_Size, 10, list);

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

void AdaBoostTrain(float pf[][15], float nf[][14], int pf1_sn, int nf1_sn, int fn, int times, int list[][2])
{
	float **q_Map = new float*[fn];

	for (int x = 0; x < fn;	x++)
		q_Map[x] = new float[3];

	//算四分位
	for (int g = 0; g < fn; g++)
	{
		//正副臉結合成一大條
		float *temp_Range = new float[pf1_sn + nf1_sn];

		for (int z = 0; z < pf1_sn; z++)
			temp_Range[z] = pf[g][z];

		for (int z = 0; z < nf1_sn; z++)
			temp_Range[z + pf1_sn] = nf[g][z];

		//cout << "!! ";
		//for (int z = 0; z < pf1_sn + nf1_sn; z++)
		//	cout << "" << temp_Range[z] << ", ";

		//cout << "\n";

		sort(temp_Range, temp_Range+pf1_sn+nf1_sn);

		//for (int z = 0; z < pf1_sn + nf1_sn; z++)
		//	cout << " " << temp_Range[z] << ", ";

		//cout << "\n";
		//system("pause");

		float q1_Index = (pf1_sn + nf1_sn + 1) / 4.0;
		float q2_Index = (pf1_sn + nf1_sn + 1) / 2.0;
		float q3_Index = (pf1_sn + nf1_sn + 1) / 4.0 * 3.0;

		//cout << "Here\n";
		//cout << q1_Index << ", " << q2_Index << ", " << q3_Index << "\n";

		if ((pf1_sn + nf1_sn + 1) % 4 == 0)
		{
			q_Map[g][0] = temp_Range[(int)q1_Index-1];			
			q_Map[g][2] = temp_Range[(int)q3_Index-1];
		}
		else
		{
			q_Map[g][0] = (temp_Range[(int)q1_Index-1] + temp_Range[(int)q1_Index])/2;		
			q_Map[g][2] = (temp_Range[(int)q3_Index-1] + temp_Range[(int)q3_Index]) / 2;
		}

		if ((pf1_sn + nf1_sn + 1) % 2 == 0)
		{
			q_Map[g][1] = temp_Range[(int)q2_Index-1];
		}
		else
		{
			q_Map[g][1] = (temp_Range[(int)q2_Index-1] + temp_Range[(int)q2_Index]) / 2;
		}
		//cout << "qIndex\n";
		//cout << (int)q1_Index << ", " << (int)q2_Index << ", " << (int)q3_Index << "\n";
	}

	//cout << "QQQQ";
	//for (int g = 0; g < fn; g++)
	//{
	//	cout << q_Map[g][0] << ", " << q_Map[g][1] << ", " << q_Map[g][2] << "\n";
	//}
	//system("pause");

	int cn2 = fn*(fn - 1) / 2;

	float *pw = new float[pf1_sn];
	float *nw = new float[nf1_sn];

	for (int i = 0; i < pf1_sn; i++)
	{
		//pw[i] = 0.5 / pf1_sn;
		pw[i] = 1.0 / (pf1_sn + nf1_sn);
	}

	for (int i = 0; i < nf1_sn; i++)
	{
		//nw[i] = 0.5 / nf1_sn;
		nw[i] = 1.0 / (pf1_sn + nf1_sn);
	}

	for (int i = 0; i < pf1_sn; i++)
	{
		cout << pw[i] << ", ";
	}

	cout << "\n";
	system("pause");

	float wsum = 0;

	for (int i = 0; i < pf1_sn; i++)
	{
		wsum = wsum + pw[i];
	}

	for (int i = 0; i < nf1_sn; i++)
	{
		wsum = wsum + nw[i];
	}

	for (int i = 0; i < pf1_sn; i++)
	{
		pw[i] /= wsum;
	}

	for (int i = 0; i < nf1_sn; i++)
	{
		nw[i] /= wsum;
	}

	float ret[2101][3];

	//OpenCL, 多傳一個List進去吧
	int pf_shape[2] = { fn, pf1_sn };
	int nf_shape[2] = { fn, nf1_sn };

	//compute.set_buffer((float *)pf, fn * pf1_sn*sizeof(float));
	//compute.set_buffer((float *)nf, fn * nf1_sn*sizeof(float));

	//compute.set_buffer((float *)pw, pf1_sn*sizeof(float));
	//compute.set_buffer((float *)nw, nf1_sn*sizeof(float));

	//compute.set_buffer((int *)pf_shape, 2 * sizeof(int));
	//compute.set_buffer((int *)nf_shape, 2 * sizeof(int));

	//compute.set_buffer(1);

	//compute.set_ret_buffer((float *)ret, fn * 3 * sizeof(float));



	for (int i = 0; i < times; i++)
	{
		float wsum = 0;

		for (int i = 0; i < pf1_sn; i++)
		{
			wsum = wsum + pw[i];
		}
		for (int i = 0; i < nf1_sn; i++)
		{
			wsum = wsum + nw[i];
		}

		for (int i = 0; i < pf1_sn; i++)
		{
			pw[i] /= wsum;
		}
		for (int i = 0; i < nf1_sn; i++)
		{
			nw[i] /= wsum;
		}



		float *output = new float[cn2];
		

		//compute.reset_buffer(2, pw);
		//compute.reset_buffer(3, nw);


		//幾個Kernel在跑
		//compute.run(2101);
		
		WeakLearn(pf, nf,pw, nw, pf1_sn, nf1_sn, fn, list, output, q_Map);

		float error = 1, theta = 0, polarity = 1;
		int selectif = -1;
		float beta;

		for (int k = 0; k < cn2; k++)
		{
			if (output[k] < error)
			{
				error = output[k];
				selectif = k;
			}
		}

		//Rebuild the error map, given k we know features


		//if (error > output[i].final_error)
		//{
		//	error = output[i].final_error;
		//	polarity = output[i].polarity;
		//	theta = output[i].theta;
		//	selectif = i;	//最好的那一"行"特徵 (從0開始算，跟Matlab誤差1)
		//}


		//printf("%f, %f, %f, %f\n", output[i].theta, output[i].polarity, output[i].final_error, selectif);
		
		
		beta = error / (1 - error);



		//計票用的4*4的16宮格
		float seatTable[4][4] = { 0 };

		//每一個資料點的座標 一對一的記錄 所以有正臉數+副臉數
		int **sn_XY = new int*[pf1_sn + nf1_sn];
		for (int k = 0; k < pf1_sn + nf1_sn; k++)
			sn_XY[k] = new int[2];

		int indX = list[selectif][0];
		int indY = list[selectif][1];

		//拿出正臉資料 取得正臉座標
		for (int z = 0; z < pf1_sn; z++)
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
		for (int z = pf1_sn; z < nf1_sn + pf1_sn; z++)
		{
			if (nf[indX][z - pf1_sn] < q_Map[indX][0])
				sn_XY[z][0] = 0;
			else if (nf[indX][z - pf1_sn] < q_Map[indX][1])
				sn_XY[z][0] = 1;
			else if (nf[indX][z - pf1_sn] < q_Map[indX][2])
				sn_XY[z][0] = 2;
			else
				sn_XY[z][0] = 3;

			if (nf[indY][z - pf1_sn] < q_Map[indY][0])
				sn_XY[z][1] = 0;
			else if (nf[indY][z - pf1_sn] < q_Map[indY][1])
				sn_XY[z][1] = 1;
			else if (nf[indY][z - pf1_sn] < q_Map[indY][2])
				sn_XY[z][1] = 2;
			else
				sn_XY[z][1] = 3;
		}

		//投票瞜
		for (int g = 0; g < pf1_sn + nf1_sn; g++)
		{
			if (g < pf1_sn)
				seatTable[sn_XY[g][0]][sn_XY[g][1]] += pw[g];
			else
				seatTable[sn_XY[g][0]][sn_XY[g][1]] -= nw[g - pf1_sn];
		}

		//正臉調權重
		for (int i = 0; i < pf1_sn; i++)
		{			
			if (seatTable[sn_XY[i][0]][sn_XY[i][1]] >= 0)
			{
				pw[i] = pw[i] * beta;
			}
		}

		//負臉調權重
		for (int i = pf1_sn; i < nf1_sn+pf1_sn; i++)
		{
			if (seatTable[sn_XY[i][0]][sn_XY[i][1]] < 0)
			{
				nw[i-pf1_sn] = nw[i-pf1_sn] * beta;
			}
		}

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
			cout << PorN(seatTable[q][0])
				<< "|" << PorN(seatTable[q][1])
				<< "|" << PorN(seatTable[q][2])
				<< "|" << PorN(seatTable[q][3]) << "\n";
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



void WeakLearn(float pf1[][15], float nf1[][14],
	float pw[], float nw[], 
	int pf1_sn, int nf1_sn, int fn, int list[][2], float* return_Matrix, float** q_Map)
{
	int cn2 = fn*(fn - 1) / 2;

	for (int h = 0; h < cn2; ++h)
	{
		
		float* one = pf1[list[h][0]];
		float* two = pf1[list[h][1]];
		float* three = nf1[list[h][0]];
		float* four = nf1[list[h][1]];
		 
		//X軸是one three

		//Y軸是two four

		float min_X = one[0];
		float max_X = one[0];
		float min_Y = two[0];
		float max_Y = two[0];

		//float max = , min = 20000000000000, error = 1, theta = 0, polarity = 1;
		
		//找最大 最小值
		for (int i = 0; i < pf1_sn; i++)
		{
			if (one[i] > max_X)
			{
				max_X = one[i];
			}
			if (one[i] < min_X)
			{
				min_X = one[i];
			}

			if (two[i] > max_Y)
			{
				max_Y = two[i];
			}
			if (two[i] < min_Y)
			{
				min_Y = two[i];
			}
		}

		for (int i = 0; i < nf1_sn; i++)
		{
			if (three[i] > max_X)
			{
				max_X = three[i];
			}
			if (three[i] < min_X)
			{
				min_X = three[i];
			}

			if (four[i] > max_Y)
			{
				max_Y = four[i];
			}
			if (four[i] < min_Y)
			{
				min_Y = four[i];
			}
		}

		//計票用的
		float seatTable[4][4] = { { 0, 0, 0, 0 },
									{ 0, 0, 0, 0 }, 
									{ 0, 0, 0, 0 }, 
									{ 0, 0, 0, 0 }};

		int **sn_XY = new int*[pf1_sn + nf1_sn];
		for (int k = 0; k < pf1_sn+nf1_sn; k++)
			sn_XY[k] = new int[2];

		//正臉
		for (int z = 0; z < pf1_sn; z++)
		{		
			if (one[z] < q_Map[list[h][0]][0])
				sn_XY[z][0] = 0;
			else if (one[z] < q_Map[list[h][0]][1])
				sn_XY[z][0] = 1;
			else if (one[z] < q_Map[list[h][0]][2])
				sn_XY[z][0] = 2;
			else
				sn_XY[z][0] = 3;

			if (two[z] < q_Map[list[h][1]][0])
				sn_XY[z][1] = 0;
			else if (two[z] < q_Map[list[h][1]][1])
				sn_XY[z][1] = 1;
			else if (two[z] < q_Map[list[h][1]][2])
				sn_XY[z][1] = 2;
			else
				sn_XY[z][1] = 3;
		}

		//負臉
		for (int z = pf1_sn; z < nf1_sn+pf1_sn; z++)
		{
			if (three[z - pf1_sn] < q_Map[list[h][0]][0])
				sn_XY[z][0] = 0;
			else if (three[z - pf1_sn] < q_Map[list[h][0]][1])
				sn_XY[z][0] = 1;
			else if (three[z - pf1_sn] < q_Map[list[h][0]][2])
				sn_XY[z][0] = 2;
			else
				sn_XY[z][0] = 3;

			if (four[z - pf1_sn] < q_Map[list[h][1]][0])
				sn_XY[z][1] = 0;
			else if (four[z - pf1_sn] < q_Map[list[h][1]][1])
				sn_XY[z][1] = 1;
			else if (four[z - pf1_sn] < q_Map[list[h][1]][2])
				sn_XY[z][1] = 2;
			else
				sn_XY[z][1] = 3;
		}

		//cout << "q_Map" << "\n";
		//for (int x = 0; x < fn; x++)
		//{
		//	cout << q_Map[x][0] << ", " << q_Map[x][1] << ", " << q_Map[x][2] << "\n";
		//}

		//for (int x = 0; x < pf1_sn + nf1_sn; x++)
		//{
		//	cout << sn_XY[x][0] << sn_XY[x][1] << "\n";
		//}

		cout << "pw, ";
		for (int j = 0; j < pf1_sn; j++)
		{
			cout << pw[j] << ", ";
		}

		cout << "\n";
		system("pause");

		//投票
		for (int g = 0; g < pf1_sn + nf1_sn; g++)
		{
			if (g < pf1_sn)
				seatTable[sn_XY[g][0]][sn_XY[g][1]] += pw[g];
			else
				seatTable[sn_XY[g][0]][sn_XY[g][1]] -= nw[g-pf1_sn];				
		}

		for (int j = 0; j < 4; j++)
		{
			cout << seatTable[j][0] << ", " << seatTable[j][1] << ", " << seatTable[j][2] << ", " << seatTable[j][3] << "\n";
		}

		system("pause");

		return_Matrix[h] = 0;

		for (int i = 0; i < pf1_sn + nf1_sn; i++)
		{
			if (i < pf1_sn)
			{
				if (seatTable[sn_XY[i][0]][sn_XY[i][1]] < 0)
					return_Matrix[h] += pw[i];
			}		
			else
			{
				if (seatTable[sn_XY[i][0]][sn_XY[i][1]] >= 0)
					return_Matrix[h] += nw[i - pf1_sn];
			}
					
		}

		printf("[%d+%d]\n", list[h][0], list[h][1]);
		for (int q = 0; q < 4; q++)
		{
			cout << PorN(seatTable[q][0])
				<< "|" << PorN(seatTable[q][1])
				<< "|" << PorN(seatTable[q][2])
				<< "|" << PorN(seatTable[q][3]) << "\n";
		}

		system("pause");

		//找最好的一刀
		//for (int j = 1; j < 10; j++)
		//{
		//	float theta1 = (max - min) / 10 * j;
		//	float error1 = 0;
		//	float polarity1 = 1;
		//	for (int i = 0; i < pf1_sn; i++)
		//	{
		//		if (pf1[i] < theta1)
		//		{
		//			error1 = error1 + pw[i];
		//		}
		//	}
		//	for (int i = 0; i < nf1_sn; i++)
		//	{
		//		if (nf1[i] > theta1)
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
		//float output[] = { error, polarity, theta };
		//return output;

		for (int i = 0; i < pf1_sn + nf1_sn; i++)
			delete[] sn_XY[i];

		delete[] sn_XY;
	}



}


