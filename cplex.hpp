#ifndef SVMWRAPPERS_cplex_hpp
#define SVMWRAPPERS_cplex_hpp

#include "problem.hpp"
#include <iostream>
#include <ilcplex/ilocplex.h>

namespace SVMWrappers
{
    namespace CPLEX
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
            IloEnv env;
            try
            {
                int numPoints = p.length();

                IloModel model(env);
                
                IloNumVar w[3];
                for(int i = 0; i < DIM; i++)
                {
                    w[i] = IloNumVar(env, -IloInfinity, IloInfinity, ILOFLOAT);
                }
                for(int i = DIM; i < 3; i++)
                {
                    w[i] = IloNumVar(env, 0, 0, ILOFLOAT);
                }

                IloNumVar b(env, -IloInfinity, IloInfinity, ILOFLOAT);

                model.add(IloMinimize(env, w[0]*w[0] + w[1]*w[1] + w[2]*w[2]/* + w[3]*w[3] + w[4]*w[4] + w[5]*w[5] + w[6]*w[6] + w[7]*w[7] + w[8]*w[8] + w[9]*w[9]*/));
                for(int i = 0; i < numPoints; i++)
                {
                    model.add(p.get_class(i)*(
                        w[0]*p.get_feature(i,0) +
                        w[1]*p.get_feature(i,1) +
                        w[2]*p.get_feature(i,2)/* +
                        w[3]*p.get_feature(i,3) +
                        w[4]*p.get_feature(i,4) +
                        w[5]*p.get_feature(i,5) +
                        w[6]*p.get_feature(i,6) +
                        w[7]*p.get_feature(i,7) +
                        w[8]*p.get_feature(i,8) +
                        w[9]*p.get_feature(i,9)*/ - b) >= 1);
                }

                // Optimize
                IloCplex cplex(model);
                cplex.setOut(env.getNullStream());
                cplex.setWarning(env.getNullStream());
                cplex.solve();
            
                if (cplex.getStatus() == IloAlgorithm::Infeasible)
                    env.out() << "No Solution" << endl;

                // env.out() << "Solution status: " << cplex.getStatus() << endl;
                // env.out() << "Optimal value: " << cplex.getObjValue() << endl;
                // env.out() << "w1: " << cplex.getValue(w[0]) << endl;
                // env.out() << "w2: " << cplex.getValue(w[1]) << endl;
                // env.out() << "w3: " << cplex.getValue(w[2]) << endl;
                // env.out() << "b: " << cplex.getValue(b) << endl;

                RowVector w_vector(1,DIM);
                for(int i = 0; i < DIM; i++)
                {
                    w_vector(0,i) = cplex.getValue(w[i]);
                }

                auto b_store = cplex.getValue(b);
                env.end();
                return Hyperplane(w_vector, b_store);
            }
            catch (IloException& ex) {
                cerr << "Error: " << ex << endl;
            }
            catch (...) {
                cerr << "Error" << endl;
            }

            Hyperplane hp;
            return hp;
        }
    }
}

#endif