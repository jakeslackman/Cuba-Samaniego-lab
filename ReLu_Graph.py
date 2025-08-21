import matplotlib.pyplot as plt
import numpy as np

# b = np.linspace(0,100,100) 
# alpha = 40 
# y = alpha - b
# for value in range(len(y)):
#     if y[value] < 0:
#         y[value] = 0

# rng = np.random.default_rng(seed=42) # reproducible randomness
# n_samples = 250                      # how many points you want
# beta_samples = rng.uniform(0, 100, n_samples)

# # ideal (noise-free) response at the sampled β’s
# y_true = alpha - beta_samples

# # add Gaussian noise (σ controls how “messy” the data look)
# sigma = 2.0
# y_noisy = y_true + rng.normal(0, sigma, n_samples)

# enforce ReLU: everything below 0 → 0
# y_noisy = np.where(y_noisy < 0, 0, y_noisy)

# # --- plot -------------------------------------------------------------------
# plt.figure(figsize=(7, 4))
# plt.plot(b,y,lw=2, label = 'ideal curve')
# plt.scatter(beta_samples, y_noisy, s=18, alpha=0.7,
#             label='samples')
# plt.xlabel('beta')
# plt.ylabel('y')
# plt.title(f'y vs beta ; alpha = {alpha}')
# plt.legend()
# plt.tight_layout()
# plt.show()

# print(y)
# plt.plot(b, y)
# plt.xlabel('beta')
# plt.ylabel('y')
# plt.title(f'plot of y as beta increases when alpha = {alpha}')
# plt.show()

#first graph, no increase in K

# import matplotlib.pyplot as plt
# import numpy as np

# b = np.linspace(0,100,100)
# alpha = 40
# k = .1
# y = 1/2 * ( (alpha - b - k)  + ((alpha - b - k)**2 + ( 4 * alpha * k))**0.5   )
# for value in range(len(y)):
#     if y[value] < 0:
#         y[value] = 0


# rng = np.random.default_rng(seed=42) # reproducible randomness
# n_samples = 250                      # how many points you want
# b_samples = rng.uniform(0, 100, n_samples)

# # ideal (noise-free) response at the sampled β’s
# y_true = 0.5 * ((alpha - b_samples - k)
#                 + np.sqrt((alpha - b_samples - k)**2 + 4*alpha*k))

#add Gaussian noise (σ controls how “messy” the data look)
# sigma = 0.0
# y_noisy = y_true + rng.normal(0, sigma, n_samples)

# # enforce ReLU: everything below 0 → 0
# y_noisy = np.where(y_noisy < 0, 0, y_noisy)

# # --- plot -------------------------------------------------------------------
# plt.figure(figsize=(7, 4))
# plt.plot(b,y,lw=2, label = 'ideal curve')

# plt.scatter(b_samples, y_noisy, s=18, alpha=0.7,
#             label='samples')
# plt.xlabel('beta')
# plt.ylabel('y')
# plt.title(f'y vs beta; alpha = {alpha}, k = {k}')
# plt.legend()
# plt.tight_layout()
# plt.show()
# plt.plot(b, y)
# plt.xlabel('beta')
# plt.ylabel('y')
# plt.title(f'plot of y as beta increases when alpha = {alpha} and epsilon = {k}')
# plt.show()
#line plot

# second graph with some K 

# alpha = np.linspace(0,400,100)
# beta = np.linspace(0,400,100)

# A,B = np.meshgrid(alpha,beta)
# line = A - B
# k=100
# y = 1/2 * ( (A - B - k)  + ((A - B - k)**2 + ( 4 * A * k))**0.5   )


# for value in range(len(line)):
#     for number in range(len(line[0])):
#         if line[value][number] < 0:
#             line[value][number] = 0

# plt.imshow(y, extent=[alpha.min(), alpha.max(), beta.min(), beta.max()], origin='lower',
#            cmap='viridis', aspect='auto')
# plt.colorbar(label=r'$y(\alpha,\beta)$')
# plt.xlabel(r'$\alpha$')
# plt.ylabel(r'$\beta$')
# plt.title(f'Heatmap of y when $k={k}$')
# plt.tight_layout()
# plt.show()

#heatmap for alpha - beta

# b = 20 
# alpha = np.linspace(0,1,10)
# phi = 1
# delta = 1
# rho, theta = 1,1
# w = .5
# y = (alpha - w) / delta
# for value in range(len(y)):
#     if y[value] < 0:
#         y[value] = 0
# beta = w

# x_axis = alpha - beta
# print(y)
# plt.plot(x_axis, y)
# plt.xlabel('alpha - beta')
# plt.ylabel('y')
# plt.title(f'plot of y vs alpha - beta as alpha increases when theta = {theta}, phi = {phi},\n delta = {delta}, rho = {rho}, and w = {w} \n assuming y can only be 0 and not negative')
# plt.show()

# #feedback with omega to 0


# alpha = np.linspace(0,1,10)
# phi = 1
# delta =1
# rho,theta = 3,3
# omega = .01
# v = rho * theta/phi
# w = -.5
# y = (1/(2*(delta + v)))* ((alpha + w - (delta * omega)) + (((alpha + w - (delta * omega))**2) + ((4*(delta + v) * alpha* omega))**.5))
# for value in range(len(y)):
#     if y[value] < 0:
#         y[value] = 0
# beta = (rho * theta * y)/phi

# x_axis = alpha - beta
# print(y)
# plt.plot(x_axis, y)
# plt.xlabel('alpha - beta')
# plt.ylabel('y')
# plt.title(f'plot of y vs alpha - beta as alpha increases when theta = {theta}, phi = {phi},\n delta = {delta}, rho = {rho}, omega = {omega}, v = {v}, w = {w} \n assuming y can only be 0 and not negative')
# plt.show()

# feedback with omega not to 0



# b = np.linspace(0,100,100)
# alpha = 20
# k = 0
# rho,theta,delta,phi = 1,1,1,1
# p_bar = (rho * theta)/(delta * phi)
# y = 1/(2 * (1+p_bar)) * ( (alpha - b - k)  + ((alpha - b - k)**2 + ( 4 * alpha * k * ( 1 + p_bar)))**0.5   )
# for value in range(len(y)):
#     if y[value] < 0:
#         y[value] = 0


# rng = np.random.default_rng(seed=42) # reproducible randomness
# n_samples = 250                      # how many points you want
# beta_samples = rng.uniform(0, 100, n_samples)

# # ideal (noise-free) response at the sampled β’s
# y_true = 1/(2 * (1+p_bar)) * ( (alpha - beta_samples - k)  + ((alpha - beta_samples - k)**2 + ( 4 * alpha * k * ( 1 + p_bar)))**0.5   )

# #add Gaussian noise (σ controls how “messy” the data look)
# sigma = 2.0
# y_noisy = y_true + rng.normal(0, sigma, n_samples)

# #enforce ReLU: everything below 0 → 0
# y_noisy = np.where(y_noisy < 0, 0, y_noisy)

# # --- plot -------------------------------------------------------------------
# plt.figure(figsize=(7, 4))
# plt.plot(b,y,lw=2, label = 'ideal curve')
# plt.scatter(beta_samples, y_noisy, s=18, alpha=0.7,
#             label='samples')
# plt.xlabel('beta')
# plt.ylabel('y')
# plt.title(f'y vs beta; alpha = {alpha}, k = {k}, rho = {rho}, theta = {theta}, phi = {phi}, delta = {delta}')
# plt.legend()
# plt.tight_layout()
# plt.show()


# plt.plot(b, y)
# plt.xlabel('beta')
# plt.ylabel('y')
# plt.title(f'plot of y as beta increases when alpha = {alpha} and epsilon = {k}')
# plt.show()
#line plot

# alpha = np.linspace(0,400,100)
# beta = np.linspace(0,400,100)

# A,B = np.meshgrid(alpha,beta)
# line = A - B

# k = 0
# rho,theta,delta,phi = 1,1,1,1
# p_bar = (rho * theta)/(delta * phi)
# y = 1/(2 * (1+p_bar)) * ( (alpha - beta - k)  + ((alpha - beta - k)**2 + ( 4 * alpha * k * ( 1 + p_bar)))**0.5   )
# for value in range(len(line)):
#     for number in range(len(line[0])):
#         if line[value][number] < 0:
#             line[value][number] = 0

# plt.imshow(y, extent=[alpha.min(), alpha.max(), beta.min(), beta.max()], origin='lower',
#            cmap='viridis', aspect='auto')

# plt.colorbar(label = '1/(2 * (1+p_bar)) * ( (alpha - b - k)  + ((alpha - b - k)**2 + ( 4 * alpha * k * ( 1 + p_bar)))**0.5)')
# plt.xlabel('Alpha')
# plt.ylabel('Beta')
# plt.title(f'Heatmap of Alpha - Beta when epison is {k}')
# # plt.show()



# alpha = np.linspace(0, 400, 100)
# beta  = np.linspace(0, 400, 100)

# A, B = np.meshgrid(alpha, beta, indexing='xy')   # 2‑D grids

# k = 100
# rho = theta = delta = phi = 1
# p_bar = (rho * theta) / (delta * phi)

# # analytic surface, computed **element‑wise** over the grids
# y = 1 / (2 * (1 + p_bar)) * (
#         (A - B - k) +
#         np.sqrt((A - B - k)**2 + 4 * A * k * (1 + p_bar))
#     )

# # ReLU‑like clipping: y < 0  →  0
# y = np.where(y < 0, 0, y)

# # ------------------------------------------------------------------
# plt.imshow(
#     y,
#     extent=[alpha.min(), alpha.max(), beta.min(), beta.max()],
#     origin='lower',
#     cmap='viridis',
#     aspect='auto'
# )
# plt.colorbar(label=r'$y(\alpha,\beta)$')
# plt.xlabel(r'$\alpha$')
# plt.ylabel(r'$\beta$')
# plt.title(f'Heatmap of y when $k={k}$, $\\bar{{p}}={p_bar}$')
# plt.tight_layout()
# plt.show()







# b = 40
# alpha = np.linspace(0,100,100)
# k = 100
# theta_2,theta_1,delta,phi = 1,.5,1,1
# theta_bar = (theta_1 * theta_2)/(delta * phi)
# y = 1/(2 * (1-theta_bar)) * ( (alpha - b - k)  + ((alpha - b - k)**2 + ( 4 * alpha * k * ( 1 - theta_bar)))**0.5   )
# for value in range(len(y)):
#     if y[value] < 0:
#         y[value] = 0


# rng = np.random.default_rng(seed=42) # reproducible randomness
# n_samples = 250                      # how many points you want
# alpha_samples = rng.uniform(0, 100, n_samples)

# # ideal (noise-free) response at the sampled β’s
# y_true = 1/(2 * (1-theta_bar)) * ( (alpha_samples - b - k)  + ((alpha_samples - b - k)**2 + ( 4 * alpha_samples * k * ( 1 - theta_bar)))**0.5   )

# #add Gaussian noise (σ controls how “messy” the data look)
# sigma = 2.0
# y_noisy = y_true + rng.normal(0, sigma, n_samples)

# #enforce ReLU: everything below 0 → 0
# y_noisy = np.where(y_noisy < 0, 0, y_noisy)

# # --- plot -------------------------------------------------------------------
# plt.figure(figsize=(7, 4))
# plt.plot(alpha,y,lw=2, label = 'ideal curve')
# plt.scatter(alpha_samples, y_noisy, s=18, alpha=0.7,
#             label='samples')
# plt.xlabel('alpha')
# plt.ylabel('y')
# plt.title(f'y vs alpha; beta = {b}, k = {k}, theta_1 = {theta_1}, theta_2 = {theta_2}, phi = {phi}, delta = {delta}')
# plt.legend()
# plt.tight_layout()
# plt.show()



# alpha = np.linspace(0, 400, 100)
# beta  = np.linspace(0, 400, 100)

# A, B = np.meshgrid(alpha, beta, indexing='xy')   # 2‑D grids

# k = 100
# theta_2 = delta = phi = 1
# theta_1 = .5
# theta_bar = (theta_1 * theta_2) / (delta * phi)

# # analytic surface, computed **element‑wise** over the grids
# y = 1 / (2 * (1 - theta_bar)) * (
#         (A - B - k) +
#         np.sqrt((A - B - k)**2 + 4 * A * k * (1 - theta_bar))
#     )

# # ReLU‑like clipping: y < 0  →  0
# y = np.where(y < 0, 0, y)

# # ------------------------------------------------------------------
# plt.imshow(
#     y,
#     extent=[alpha.min(), alpha.max(), beta.min(), beta.max()],
#     origin='lower',
#     cmap='viridis',
#     aspect='auto'
# )
# plt.colorbar(label=r'$y(\alpha,\beta)$')
# plt.xlabel(r'$\alpha$')
# plt.ylabel(r'$\beta$')
# plt.title(f'Heatmap of y when $k={k}$, $\\bar{{θ}}={theta_bar}$')
# plt.tight_layout()
# plt.show()
n_samples=5_000
k=0
alpha_range=(0.1, 2.0)
beta_range=(0.1, 2.0)
diff_limits=(-1, 1)
    # 1 . draw α and β independently
alpha = np.random.uniform(*alpha_range, n_samples)
beta  = np.random.uniform(*beta_range,  n_samples)

# 2 . difference we care about
diff  = alpha - beta

# 3 . keep only points with −1 ≤ diff ≤ 1
mask  = (diff >= diff_limits[0]) & (diff <= diff_limits[1])
alpha_f = alpha[mask]
beta_f  = beta[mask]
diff_f  = diff[mask]

# 4 . evaluate your analytic expression
# y = 0.5 * ((alpha_f - beta_f - k) +
#             np.sqrt((alpha_f - beta_f - k)**2 + 4 * alpha_f * k))
# rho,theta,delta,phi = 1,0,1,1
# p_bar = (rho * theta)/(delta * phi)
# y = 1/(2 * (1+p_bar)) * ( (alpha_f - beta_f - k)  + ((alpha_f - beta_f - k)**2 + ( 4 * alpha_f * k * ( 1 + p_bar)))**0.5   )
theta_2,theta_1,delta,phi = 1,1.1,1,1
theta_bar = (theta_1 * theta_2)/(delta * phi)
y = 1/(2 * (1-theta_bar)) * ( (alpha_f - beta_f - k)  - ((alpha_f - beta_f - k)**2 + ( 4 * alpha_f * k * ( 1 - theta_bar)))**0.5   )

# 5 . scatter plot
plt.figure(figsize=(6, 4))
plt.scatter(diff_f, y, s=10)
plt.xlabel(r'$\alpha - \beta$')
plt.ylabel('y')
plt.title(f'Scatter for α – β, k = {k}, theta_1 = {theta_1}, theta_2 = {theta_2}')
plt.xlim(diff_limits)
plt.tight_layout()
plt.show()

#sample_and_plot_random_b()