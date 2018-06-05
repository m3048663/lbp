
#include "./liblinear-1.96/linear.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define MODEL "/home/ubuntu/lbp/svm/model"

void read_txt(char* filename, float* lbp)
{
	
	int i = 0;
	FILE *fp;
	fp = fopen(filename,"r");
	if (!fp)
		printf("please check the reading file\n");
	else
		while (!feof(fp))
			{
				if (fscanf(fp, "%f", &lbp[i]) != 1)
					break;
				i++;
				fgetc(fp);
			}
}




int main()
{	
	float y[59];
	float cb[59];
	float cr[59];
	read_txt("../false_y.txt", y);
	read_txt("../false_cb.txt", cb);
	read_txt("../false_cr.txt", cr);
	feature_node* x = (feature_node*)malloc(sizeof(feature_node) * 177);
	//struct feature_node x[177];
	
	for(int j = 0; j < 59; j++)	
	{
		x[j].index = j;
		x[j].value = y[j];
		//printf("%f, ",lbp[j]);
	}
	for(int j = 59; j < 118; j++)	
	{
		x[j].index = j;
		x[j].value = cb[j - 59];
		//printf("%f, ",lbp[j]);
	}
	for(int j = 118; j < 177; j++)	
	{
		x[j].index = j;
		x[j].value = cr[j - 118];
		//printf("%f, ",lbp[j]);
	}

	model *mod = new model;
	mod = load_model(MODEL);

	double p = 0.0f;
	p = predict(mod, x);	
	std::cout << "p = " << p << std::endl;

	free_model_content(mod);
	return 0;
}
