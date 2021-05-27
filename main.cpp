#include "problem.hpp"
#include "libsvm.hpp"
#include "cplex.hpp"
#include "Eigen/Dense"
#include <ctime>
#include <ratio>
#include <chrono>

using namespace std::chrono;
using RowVector = Eigen::Matrix<double, 1, Eigen::Dynamic>;
using Hyperplane = Eigen::Hyperplane<double, Eigen::Dynamic>;

double average(int* arr, int size)
{
    double sum = 0;
    for(int i = 0; i < size; i++)
    {
        sum+= arr[i];
    }

    return sum/size;
}

double st_dev(int* arr, int size, double average)
{
    double sumSquares = 0;
    for(int i = 0; i < size; i++)
    {
        sumSquares += pow(arr[i] - average, 2);
    }

    return sqrt(sumSquares / size);
}

int main()
{
    SVMWrappers::Problem<double, 2> p(8);
    SVMWrappers::Problem<double, 3> p2(16);
    RowVector rv(1,2);
    RowVector rv2(1,3);
    Hyperplane hp;
    double randx1;
    double randx2;
    double randy1;
    double randy2;
    double randz1;
    double randz2;
    int distancex;
    int distancey;
    int distancez;
    int numTrials = 1000;
    int cplexLongestSquare = -1;
    int libsvmLongestSquare = -1;
    int cplexLongestCube = -1;
    int libsvmLongestCube = -1;
    int* times = new int[numTrials];

    // 2d test
    for(int i = 0; i < numTrials; i++)
    {
        
        while(true)
        {
            randx1 = (rand() % 10000) / 100;
            randx2 = (rand() % 10000) / 100;
            distancex = abs(randx1-randx2);
            randy1 = (rand() % 10000) / 100;
            randy2 = (rand() % 10000) / 100;
            distancey = abs(randy1-randy2);
            if(distancex > 0 && distancey > 0)
                break;
        }
    
        double size = (rand() % (min(distancex, distancey) * 100)) / 100;
        if(size > 0.01)
            size -= 0.01;
        
        rv(0,0) = randx1 - size/2; // bot left corner
        rv(0,1) = randy1 - size/2;
        p.set_vector(0, rv, 1);
        rv(0,1) = randy1 + size/2; // top left
        p.set_vector(1, rv, 1);
        rv(0,0) = randx1 + size/2; // top right
        p.set_vector(3, rv, 1);
        rv(0,1) = randy1 - size/2; // bot right
        p.set_vector(2, rv, 1);

        rv(0,0) = randx2 - size/2; // bot left corner
        rv(0,1) = randy2 - size/2;
        p.set_vector(4, rv, -1);
        rv(0,1) = randy2 + size/2; // top left
        p.set_vector(5, rv, -1);
        rv(0,0) = randx2 + size/2; // top right
        p.set_vector(7, rv, -1);
        rv(0,1) = randy2 - size/2; // bot right
        p.set_vector(6, rv, -1);

        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        hp = SVMWrappers::CPLEX::solve(p);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double, std::milli> time_span = t2 - t1;
        if(time_span.count() > cplexLongestSquare)
            cplexLongestSquare = time_span.count();

        std::cout << hp.normal() << " " << i << std::endl;
        times[i] = time_span.count();
    }
    double cplexAverageSquare = average(times, numTrials);
    double cplexStDevSquare = st_dev(times, numTrials, cplexAverageSquare);
    
    for(int i = 0; i < numTrials; i++)
    {
        
        while(true)
        {
            randx1 = (rand() % 10000) / 100;
            randx2 = (rand() % 10000) / 100;
            distancex = abs(randx1-randx2);
            randy1 = (rand() % 10000) / 100;
            randy2 = (rand() % 10000) / 100;
            distancey = abs(randy1-randy2);
            if(distancex > 0 && distancey > 0)
                break;
        }
    
        double size = (rand() % (min(distancex, distancey) * 100)) / 100;
        if(size > 0.01)
            size -= 0.01;
        
        rv(0,0) = randx1 - size/2; // bot left corner
        rv(0,1) = randy1 - size/2;
        p.set_vector(0, rv, 1);
        rv(0,1) = randy1 + size/2; // top left
        p.set_vector(1, rv, 1);
        rv(0,0) = randx1 + size/2; // top right
        p.set_vector(3, rv, 1);
        rv(0,1) = randy1 - size/2; // bot right
        p.set_vector(2, rv, 1);

        rv(0,0) = randx2 - size/2; // bot left corner
        rv(0,1) = randy2 - size/2;
        p.set_vector(4, rv, -1);
        rv(0,1) = randy2 + size/2; // top left
        p.set_vector(5, rv, -1);
        rv(0,0) = randx2 + size/2; // top right
        p.set_vector(7, rv, -1);
        rv(0,1) = randy2 - size/2; // bot right
        p.set_vector(6, rv, -1);

        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        hp = SVMWrappers::LIBSVM::solve(p);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double, std::milli> time_span = t2 - t1;
        if(time_span.count() > libsvmLongestSquare)
            libsvmLongestSquare = time_span.count();

        std::cout << hp.normal() << " " << i << std::endl;
        times[i] = time_span.count();
    }

    double libsvmAverageSquare = average(times, numTrials);
    double libsvmStDevSquare = st_dev(times, numTrials, libsvmAverageSquare);
    
    // 3d test
    for(int i = 0; i < numTrials; i++)
    {
        while(true)
        {
            randx1 = (rand() % 10000) / 100;
            randx2 = (rand() % 10000) / 100;
            distancex = abs(randx1-randx2);
            randy1 = (rand() % 10000) / 100;
            randy2 = (rand() % 10000) / 100;
            distancey = abs(randy1-randy2);
            randz1 = (rand() % 10000) / 100;
            randz2 = (rand() % 10000) / 100;
            distancez = abs(randz1-randz2);
            if(distancex > 0 && distancey > 0 && distancez > 0)
                break;
        }
    
        double size = (rand() % (min(distancez, min(distancex, distancey)) * 100)) / 100;
        if(size > 0.01)
            size -= 0.01;
        
        rv2(0,0) = randx1 - size/2; // bot left front corner
        rv2(0,1) = randy1 - size/2;
        rv2(0,2) = randz1 - size/2;
        p2.set_vector(0, rv2, 1);
        rv2(0,2) = randz1 + size/2; // bot left back
        p2.set_vector(1, rv2, 1);
        rv2(0,0) = randx1 + size/2; // bot right back
        p2.set_vector(2, rv2, 1);
        rv2(0,2) = randz1 - size/2; // bot right front
        p2.set_vector(3, rv2, 1);
        rv2(0,1) = randy1 + size/2; // top right front
        p2.set_vector(4, rv2, 1);
        rv2(0,0) = randx1 - size/2; // top left front
        p2.set_vector(5, rv2, 1);
        rv2(0,2) = randz1 + size/2; // top left back
        p2.set_vector(6, rv2, 1);
        rv2(0,0) = randx1 + size/2; // top right back
        p2.set_vector(7, rv2, 1);

        rv2(0,0) = randx2 - size/2; // bot left front corner
        rv2(0,1) = randy2 - size/2;
        rv2(0,2) = randz2 - size/2;
        p2.set_vector(8, rv2, -1);
        rv2(0,2) = randz2 + size/2; // bot left back
        p2.set_vector(9, rv2, -1);
        rv2(0,0) = randx2 + size/2; // bot right back
        p2.set_vector(10, rv2, -1);
        rv2(0,2) = randz2 - size/2; // bot right front
        p2.set_vector(11, rv2, -1);
        rv2(0,1) = randy2 + size/2; // top right front
        p2.set_vector(12, rv2, -1);
        rv2(0,0) = randx2 - size/2; // top left front
        p2.set_vector(13, rv2, -1);
        rv2(0,2) = randz2 + size/2; // top left back
        p2.set_vector(14, rv2, -1);
        rv2(0,0) = randx2 + size/2; // top right back
        p2.set_vector(15, rv2, -1);

        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        hp = SVMWrappers::CPLEX::solve(p2);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double, std::milli> time_span = t2 - t1;
        if(time_span.count() > cplexLongestCube)
            cplexLongestCube = time_span.count();

        std::cout << hp.normal() << " " << i << std::endl;
        times[i] = time_span.count();
    }
    double cplexAverageCube = average(times, numTrials);
    double cplexStDevCube = st_dev(times, numTrials, cplexAverageCube);

    for(int i = 0; i < numTrials; i++)
    {
        while(true)
        {
            randx1 = (rand() % 10000) / 100;
            randx2 = (rand() % 10000) / 100;
            distancex = abs(randx1-randx2);
            randy1 = (rand() % 10000) / 100;
            randy2 = (rand() % 10000) / 100;
            distancey = abs(randy1-randy2);
            randz1 = (rand() % 10000) / 100;
            randz2 = (rand() % 10000) / 100;
            distancez = abs(randz1-randz2);
            if(distancex > 0 && distancey > 0 && distancez > 0)
                break;
        }
    
        double size = (rand() % (min(distancez, min(distancex, distancey)) * 100)) / 100;
        if(size > 0.01)
            size -= 0.01;
        
        rv2(0,0) = randx1 - size/2; // bot left front corner
        rv2(0,1) = randy1 - size/2;
        rv2(0,2) = randz1 - size/2;
        p2.set_vector(0, rv2, 1);
        rv2(0,2) = randz1 + size/2; // bot left back
        p2.set_vector(1, rv2, 1);
        rv2(0,0) = randx1 + size/2; // bot right back
        p2.set_vector(2, rv2, 1);
        rv2(0,2) = randz1 - size/2; // bot right front
        p2.set_vector(3, rv2, 1);
        rv2(0,1) = randy1 + size/2; // top right front
        p2.set_vector(4, rv2, 1);
        rv2(0,0) = randx1 - size/2; // top left front
        p2.set_vector(5, rv2, 1);
        rv2(0,2) = randz1 + size/2; // top left back
        p2.set_vector(6, rv2, 1);
        rv2(0,0) = randx1 + size/2; // top right back
        p2.set_vector(7, rv2, 1);

        rv2(0,0) = randx2 - size/2; // bot left front corner
        rv2(0,1) = randy2 - size/2;
        rv2(0,2) = randz2 - size/2;
        p2.set_vector(8, rv2, -1);
        rv2(0,2) = randz2 + size/2; // bot left back
        p2.set_vector(9, rv2, -1);
        rv2(0,0) = randx2 + size/2; // bot right back
        p2.set_vector(10, rv2, -1);
        rv2(0,2) = randz2 - size/2; // bot right front
        p2.set_vector(11, rv2, -1);
        rv2(0,1) = randy2 + size/2; // top right front
        p2.set_vector(12, rv2, -1);
        rv2(0,0) = randx2 - size/2; // top left front
        p2.set_vector(13, rv2, -1);
        rv2(0,2) = randz2 + size/2; // top left back
        p2.set_vector(14, rv2, -1);
        rv2(0,0) = randx2 + size/2; // top right back
        p2.set_vector(15, rv2, -1);

        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        hp = SVMWrappers::LIBSVM::solve(p2);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double, std::milli> time_span = t2 - t1;
        if(time_span.count() > libsvmLongestCube)
            libsvmLongestCube = time_span.count();

        std::cout << hp.normal() << " " << i << std::endl;
        times[i] = time_span.count();
    }
    double libsvmAverageCube = average(times, numTrials);
    double libsvmStDevCube = st_dev(times, numTrials, libsvmAverageCube);
    
    std::cout << "\nSquare test\n";
    std::cout << "Cplex average: " << cplexAverageSquare << std::endl;
    std::cout << "Cplex longest: " << cplexLongestSquare << std::endl;
    std::cout << "Cplex st dev: " << cplexStDevSquare << std::endl;
    std::cout << "Libsvm average: " << libsvmAverageSquare << std::endl;
    std::cout << "Libsvm longest: " << libsvmLongestSquare << std::endl;
    std::cout << "Libsvm st dev: " << libsvmStDevSquare << std::endl;

    std::cout << "\nCube test\n";
    std::cout << "Cplex average: " << cplexAverageCube << std::endl;
    std::cout << "Cplex longest: " << cplexLongestCube << std::endl;
    std::cout << "Cplex st dev: " << cplexStDevCube << std::endl;
    std::cout << "Libsvm average: " << libsvmAverageCube << std::endl;
    std::cout << "Libsvm longest: " << libsvmLongestCube << std::endl;
    std::cout << "Libsvm st dev: " << libsvmStDevCube << std::endl;
    
    delete[] times;
    return 0;
} 
