#ifndef SVMWRAPPERS_libsvm_hpp
#define SVMWRAPPERS_libsvm_hpp

#include "problem.hpp"
#include <iostream>
#include "libsvm-3.24/svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

namespace SVMWrappers
{
    namespace LIBSVM
    {
        using namespace std;
        template<typename T, unsigned int DIM>
        Eigen::Hyperplane<T, Eigen::Dynamic> solve(Problem<T, DIM> p)
        {
            using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
            using RowVector = Eigen::Matrix<T, 1, Eigen::Dynamic>;
            using Index = Eigen::Index;
            using Hyperplane = Eigen::Hyperplane<T, Eigen::Dynamic>;

            struct svm_parameter param;
            struct svm_problem prob;
            struct svm_model *model;
            struct svm_node *x_space;
            size_t elements = 0;

            // determine how many elements are needed
            for(int i = 0; i < p.length(); ++i)
            {
                for(int j = 0; j < p.dimension(); ++j)
                {
                    if(p.get_feature(i, j) != 0)
                        ++elements;
                }
                ++elements;
            }

            // set parameters
            param.svm_type = 0;
            param.kernel_type = 0;
            param.degree = 3;
            param.gamma = 0;
            param.coef0 = 0;
            param.nu = 0.5;
            param.cache_size = 100;
            param.C = 99999;
            param.eps = 1e-3;
            param.p = 0.1;
            param.shrinking = 1;
            param.probability = 0;
            param.nr_weight = 0;
            param.weight_label = NULL;
            param.weight = NULL;

            // read problem
            prob.l = p.length();
            prob.y = Malloc(double, prob.l);
	        prob.x = Malloc(struct svm_node *, prob.l);
            x_space = Malloc(struct svm_node, elements);

            int max_index = -1;
            int currentElement = 0;
            for(int i = 0; i < prob.l; ++i)
            {
                prob.y[i] = p.get_class(i);
                prob.x[i] = &x_space[currentElement];
                for(int j = 0; j < p.dimension(); ++j)
                {
                    if(p.get_feature(i, j) != 0)
                    {
                        x_space[currentElement].index = j + 1;
                        x_space[currentElement].value = p.get_feature(i, j);
                        ++currentElement; 
                        if(j+1 > max_index)
                            max_index = j+1;
                    }
                }
                x_space[currentElement].index = -1;
                ++currentElement; 
            }
            if(max_index > 0)
            {
                param.gamma = 1.0 / max_index;
            }

            model = svm_train(&prob,&param);

            // coefficients are in coefs[0][k] where k is support vector number
            double** coefs = model->sv_coef;

            // svm coordinates are in svs[k][i] where k is support vector number and i is feature number
            svm_node** svs = model->SV;

            int* nsv = model->nSV;
            RowVector w(1,DIM);
            for(int i = 0; i < DIM; i++)
            {
                double total = 0;
                for(int j = 0; j < nsv[0] + nsv[1]; j++)
                {
                    total += svs[j][i].value * coefs[0][j];
                }
                w(0,i) = total;
            }
            return Hyperplane(w, -1 * model->rho[0]);
        }
    }
}

#endif