import matplotlib.pyplot as plt

bayesian = [9.577905099631288,  5.97*10**(-4)]
PC = [3.087697736642583, 6.45*10**(-4)]
EI = [2.8875151246913853, 149.13*10**(-4)]
CI = [1.8168054450729332, 6.64*10**(-4)]

a = [bayesian, PC, EI, CI]
truth=[3, 0]

for b in a:
    c = 0
    for d, e in zip(b, truth):
        c += (d - e)**2
    print(c)


ANEES = [9.577905099631288, 3.087697736642583, 2.8875151246913853, 1.8168054450729332]
MSE = [5.97*10**(-4), 6.45*10**(-4), 149.13*10**(-4), 6.64*10**(-4)]

l = 1
ax = plt.axes()
plt.ylim([-0.001, 160*10**(-4)])
plt.xlim([1.5, 11])
plt.scatter(3, 0, color='black', zorder=50)
plt.annotate("truth", (3, -0.00075))

plt.scatter(bayesian[0], bayesian[1])
plt.annotate("bayesian", (bayesian[0], bayesian[1]+0.0005))
plt.plot([bayesian[0], truth[0]], [bayesian[1], truth[1]], linestyle='dashed', linewidth=l, zorder=10)

plt.scatter(PC[0], PC[1])
plt.annotate("PC (0.05)", (PC[0]+0.1, PC[1]))
plt.plot([PC[0], truth[0]], [PC[1], truth[1]], linestyle='dashed', linewidth=1, zorder=10)

plt.scatter(EI[0], EI[1], label="EI")
plt.annotate("EI", (EI[0]+0.1, EI[1]))
plt.plot([EI[0], truth[0]], [EI[1], truth[1]], linestyle='dashed', linewidth=l, zorder=10)

plt.scatter(CI[0], CI[1])
plt.annotate("CI", (CI[0], CI[1]+0.0005))
plt.plot([CI[0], truth[0]], [CI[1], truth[1]], linestyle='dashed', linewidth=l, zorder=10)

plt.xlabel('ANEES')
plt.ylabel('MSE')


plt.show()