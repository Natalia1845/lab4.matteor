import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Дано для акцій A1, A2, A3
expected_returns = np.array([0.10, 0.20, 0.50])  # Сподівані норми прибутку
std_devs = np.array([0.02, 0.10, 0.20])  # Середньоквадратичні відхилення
correlations = np.array([[1, 0, 0], [0, 1, -0.6], [0, -0.6, 1]])  # Коефіцієнти кореляції

# Обчислення ковариаційної матриці
cov_matrix = np.outer(std_devs, std_devs) * correlations

# Функція для розрахунку сподіваної норми прибутку портфеля
def expected_return(weights, expected_returns):
    return np.dot(weights, expected_returns)

# Функція для розрахунку варіансу (ризику) портфеля
def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

# Функція для розрахунку детермінованого еквівалента (minimization)
def objective(weights, expected_returns, cov_matrix, m=0.30):
    # Сподівана норма прибутку та варіанс портфеля
    ret = expected_return(weights, expected_returns)
    risk = portfolio_variance(weights, cov_matrix)
    # Вибір для задачі одержання бажаного прибутку
    return (ret - m) ** 2 + risk

# Вага портфеля (початкові значення)
initial_weights = np.array([1/3, 1/3, 1/3])

# Обмеження: сума ваг повинна бути рівною 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Границі ваг (всі ваги мають бути між 0 і 1)
bounds = [(0, 1), (0, 1), (0, 1)]

# Мінімізуємо функцію для задачі одержання бажаного прибутку (m = 30%)
result = minimize(objective, initial_weights, args=(expected_returns, cov_matrix, 0.30), bounds=bounds, constraints=constraints)

# Оптимальні ваги для портфеля
optimal_weights = result.x

# Сподівана норма прибутку та ризик для оптимального портфеля
optimal_return = expected_return(optimal_weights, expected_returns)
optimal_risk = np.sqrt(portfolio_variance(optimal_weights, cov_matrix))

# Виведення результатів
print("Оптимальні ваги портфеля:", optimal_weights)
print("Сподівана норма прибутку портфеля:", optimal_return)
print("Ризик (стандартне відхилення) портфеля:", optimal_risk)

# Побудова множини допустимих та ефективних портфелів
# Генеруємо 1000 різних портфелів
num_portfolios = 1000
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    weights = np.random.random(3)
    weights /= np.sum(weights)  # Нормалізація, щоб сума ваг була рівною 1
    portfolio_ret = expected_return(weights, expected_returns)
    portfolio_risk = np.sqrt(portfolio_variance(weights, cov_matrix))
    results[0,i] = portfolio_ret
    results[1,i] = portfolio_risk
    results[2,i] = portfolio_ret / portfolio_risk  # Ризикована норма прибутку

# Візуалізація ефективного фронту
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='YlGnBu', marker='o')
plt.title('Множина допустимих та ефективних портфелів')
plt.xlabel('Ризик (стандартне відхилення)')
plt.ylabel('Сподівана норма прибутку')
plt.colorbar(label='Ризикована норма прибутку')
plt.show()
