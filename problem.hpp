#ifndef SVM_PROBLEM_HPP
#define SVM_PROBLEM_HPP

#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include "Eigen/Geometry"
#include <limits>
#include <iostream>

using namespace std;

namespace SVMWrappers {
template<typename T, unsigned int DIM>
    class Problem {
        public:
            using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
            using RowVector = Eigen::Matrix<T, 1, Eigen::Dynamic>;
            using Index = Eigen::Index;
            using Hyperplane = Eigen::Hyperplane<T, Eigen::Dynamic>;

            // create an svm problem with n vectors
            Problem(Index N = 0):
                    vectors_mtr(N, DIM),
                    classes(N)
            {
            }

            // set a vector
            void set_vector(Index vector_idx, const RowVector& values, T classification)
            {
                vectors_mtr.row(vector_idx) = values;
                classes(vector_idx) = classification;
            }

            // add new vector
            void add_vector(const RowVector& values, T classification)
            {
                // resize the vectors matrix and the classifications vector
                vectors_mtr.conservativeResize(vectors_mtr.rows() + 1, Eigen::NoChange);
                classes.conservativeResize(classes.rows() + 1, Eigen::NoChange);
                
                vectors_mtr.row(vectors_mtr.rows() - 1) = values;
                classes(classes.rows() - 1) = classification;
            }

            RowVector get_vector(Index index)
            {
                return vectors_mtr.row(index);
            }

            T get_class(Index index)
            {
                return classes(index);
            }

            T get_feature(Index row, Index column)
            {
                if(row > vectors_mtr.rows() || column > vectors_mtr.cols())
                    return 0;
                return vectors_mtr(row, column);
            }

            int length()
            {
                return vectors_mtr.rows();
            }

            int dimension()
            {
                return vectors_mtr.cols();
            }

            void print()
            {
                cout << vectors_mtr << endl << classes << endl;
            }

        private:
            Matrix vectors_mtr;
            Vector classes;
    };
}
#endif