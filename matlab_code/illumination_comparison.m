Phi = 1; % Световой поток источника
N = 2000; % Количество лучей
T = 5; % Количество соударений
r = linspace(0.04, 5, 100); % Диапазон значений r
rho = ones(1, N + 1) * 0.5; % Коэффициенты отражения для всех лучей 0.5

% Засекаем время для локальной оценки
tic;
illumination_local = zeros(size(r));
for i = 1:length(r)
    illumination_local(i) = localEstimate(Phi, N, T, rho, r(i));
end
local_time = toc;

h1 = 0.5;
h2 = 0.5;
rho1 = 0.5;
rho2 = 0.5;

% Засекаем время для метода Соболева
tic;
illuminationSobolev = zeros(size(r));
for i = 1:length(r)
    result = h1 / ((h1^2 + r(i)^2)^1.5);
    for k = 0.01:0.01:10
        term1 = exp(-h1 * k) * rho1 * k * besselk(1,k)+exp(-h2 * k);
        term2 = 1 - rho1 * rho2 * k^2 * besselk(1,k)^2;
        term3 = besselk(1,k) * besselj(0, k * r(i)) * k^2;
        result = result + term1 / term2 * term3 * 0.01; % ? dk
    end
    illuminationSobolev(i) = result;
end
sobolev_time = toc;

% Выводим результаты
fprintf('Время выполнения алгоритма локальной оценки: %.4f секунд\n', local_time);
fprintf('Время выполнения алгоритма метода Соболева: %.4f секунд\n', sobolev_time);
fprintf('Алгоритм локальной оценки выполняется в %.2f раз быстрее, чем метод Соболева.\n', sobolev_time/local_time);


figure;
plot(r, illuminationSobolev, 'LineWidth', 2);

hold on;
plot(r, illumination_local, 'LineWidth', 2);
hold off;

xlabel('Расстояние r');
ylabel('Освещенность ');
title('График распределения освещенности по плоскости');
grid on;


function result = localEstimate(Phi, N, T, rho, r)
    result = 0;
    pi_val = pi;

    % Вычисление веса каждого луча Q
    Q = ones(1, N + 1); % Изначально у всех 1
    for i = 2:N+1
        Q(i) = Q(i - 1) * rho(i-1); % Умножаем на коэффицент отражения (0.5), т.е. уменьшаем каждый следующий в 2 раза
    end

    % Вычисление освещенности для каждого луча и каждого соударения
    for n = 1:N
        for i = 1:T+1
            F = 1 / (pi_val * r); % ? Функция F обратно пропорциональная расстоянию
            result = result + Phi / (pi_val * N) * Q(i) * F;
        end
    end
end

