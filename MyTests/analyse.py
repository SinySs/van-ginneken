import matplotlib.pyplot as plt

test_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9,  10]
L_values = [n * 100 for n in test_numbers]  # L = номер теста * 100
RAT_values = [895.5600, 794.3100, 693.0600, 591.8100, 490.5600, 
              389.3550, 288.1650, 186.9350, 85.6850, -15.5650]
time_values = [0.1853, 0.4227, 0.7831, 1.4028, 2.1579, 
               3.0246, 3.8752, 4.4927, 4.8231, 5.2310]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(L_values, time_values, 'bo-', label='Время выполнения')
plt.xlabel('Длина L (номер теста × 100)')
plt.ylabel('Время выполнения (сек)')
plt.title('Зависимость времени выполнения от длины')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(L_values, RAT_values, 'ro-', label='Значение RAT')
plt.xlabel('Длина L (номер теста × 100)')
plt.ylabel('Значение RAT')
plt.title('Зависимость RAT от длины')
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig('analysis_results.png', dpi=300)

plt.show()