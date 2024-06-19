#include <iostream>
#include <cmath>
#include <boost/math/special_functions/bessel.hpp>
#include <chrono>


using namespace std;
using namespace boost::math;
using namespace chrono;


double sobolevTask(double h1, double h2, double rho1, double rho2, double r) {
    double result = 0.0;
    double k_max = 10000.0; // диапазон интегрирования

    // Первая часть
    result += h1 / pow((h1 * h1 + r * r), 1.5);

    // Вторая часть, интеграл
    for (double k = 0.01; k <= k_max; k += 0.01) {
        double term1 = exp(-h1 * k) * rho1 * k * cyl_bessel_k(1, k) + exp(-h2 * k);
        double term2 = 1 - rho1 * rho2 * k * k * cyl_bessel_k(1, k) * cyl_bessel_k(1, k);
        result += (term1 / term2) * cyl_bessel_k(1, k) * cyl_bessel_j(0, k * r) * k * k;
    }

    return result;
}

double localEstimate(double Phi, int N, int T, double rho1, double rho2, double r) {
    double result = 0.0;
    double pi = 3.14159265358979323846;

    // Вычисляем вес луча Q
    double Q = rho1 * rho2;

    // Вычисляем освещенность для каждого луча
    for (int n = 1; n <= N; ++n) {
        for (int i = 0; i <= T; ++i) {
            
            double h1 = 0.5; // Расстояние от 1-ой плоскости до источника
            double F = h1 / pow((h1 * h1 + r * r), 1.5); // F(r, r_si) ?

            result += Q * F;
        }
    }

    // Нормализуем освещенность
    result *= Phi / (pi * N);

    return result;
}


int main() {
    // Параметры плоскости и источника света
    double h1 = 0.5; // Расстояние от 1-ой плоскости до источника
    double h2 = 0.5; // Расстояние от 1-ой плоскости до источника
    double rho1 = 0.5; // Коэффициент отражения 1-ой плоскости
    double rho2 = 0.5; // Коэффициент отражения 2-ой плоскости
    double r = 1.0; // Удаленность точки на плоскости от источника

    auto start_sobolev = high_resolution_clock::now();
    double sobolevResult = sobolevTask(h1, h2, rho1, rho2, r);
    auto end_sobolev = high_resolution_clock::now();
    auto duration_sobolev = duration_cast<microseconds>(end_sobolev - start_sobolev);
    cout << "Распределение облученности на плоскости по формуле Соболева: " << sobolevResult << endl;
    cout << "Время выполнения: " << duration_sobolev.count() << " микросекунд" << endl;

    
    double Phi = 1.0; // Световой поток источника
    int N = 2000; // Количество лучей
    int T = 5; // Количество соударений

    auto start_local = high_resolution_clock::now();
    double localResult = localEstimate(Phi, N, T, rho1, rho2, r);
    auto end_local = high_resolution_clock::now();
    auto duration_local = duration_cast<microseconds>(end_local - start_local);
    cout << "Распределение облученности на плоскости по формуле локальной оценки: " << localResult << endl;
    cout << "Время выполнения функции: " << duration_local.count() << " микросекунд" << endl;


    return 0;
}
