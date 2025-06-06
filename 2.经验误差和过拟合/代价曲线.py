import numpy as np
import matplotlib.pyplot as plt

# 模型 A: FPR=0.3, FNR=0.4
FPR_A, FNR_A = 0.3, 0.4
# 模型 B: FPR=0.2, FNR=0.6
FPR_B, FNR_B = 0.2, 0.6

pcf = np.linspace(0, 1, 100)
cost_A = pcf * FNR_A + (1 - pcf) * FPR_A
cost_B = pcf * FNR_B + (1 - pcf) * FPR_B

plt.figure(figsize=(6, 5))
plt.plot(pcf, cost_A, label='Model A: EC_A(c)', color='blue')
plt.fill_between(pcf, cost_A, color='lightblue', alpha=0.4, label='Area A = (FPR_A + FNR_A)/2')
plt.plot(pcf, cost_B, label='Model B: EC_B(c)', color='green')
plt.fill_between(pcf, cost_B, color='lightgreen', alpha=0.4, label='Area B = (FPR_B + FNR_B)/2')
plt.xlabel('Probability Cost Function (PCF)')
plt.ylabel('Expected Normalized Cost')
plt.title('Comparison of Two Models on Cost Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
