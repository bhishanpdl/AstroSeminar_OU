# PartIII-Astrostatistics
Home Page for Astrostatistics Course, Part III Mathematics / Part III Astrophysics

Lent Term
Tuesday, Thursday & Saturday at 12 noon. CMS **Meeting Room 5**.

**Final Exam: Monday, 11 June 2018, 1:30 pm to 4:30 pm.  Closed book, answer 3 of 4 equal-weight questions.**

**If you did not receive an annoucement email from me, then you are not on my email list.  If you want to be on the list, please let me know.**  

Office Hours: Fridays @ 1pm  
Statistical Laboratory  
CMS Pavilion D, Office 1.07  

Recommended Texts:  
(Both texts are freely available through the Cambridge Library website.)

F&B = Feigelson & Babu. "Modern Statistical Methods for Astronomy"  
Ivezić = Ivezić, Connolly, VanderPlas & Gray. "Statistics, Data Mining, and Machine Learning in Astronomy"

**Week 1**  
Lecture 01 (Thu 18 Jan) has been uploaded!
  * Introduction to Astrostatistics
  * Introduction to Case Studies

Lecture 02 (Sat 20 Jan) has been uploaded!
  * Introduction to Astronomical Data Types
  * Overview of Case Study on Modelling Stellar Spectra with Gaussian Processes
  * Reference: Czekala et al. 2017, The Astrophysical Journal, 840 49

Lecture 03 (Tue 23 Jan) covered material on probability from  
  * Feigelson & Babu (F & B): Chapter 2, or 
  * Ivezić: Chapter 3 (through Ch 3.1.3)

**Week 2**  
Lecture 04 (Thu 25 Jan) slides uploaded. Covered:
  * Limit theorems and started statistical inference (up to maximum likelihood)
  * F & B: finished Ch 2, start Ch 3, or
  * Ivezić: finished Ch 3, start Ch 4 
  
Lecture 05 (Sat 27 Jan) slides uploaded.  Covered:
  * Frequentist properties of estimators (unbiasedness, MSE, consistency, asymptotics)
  * Maximum likelihood estimators, frequentist properties, Fisher Matrix and Cramer-Rao bound
  * F & B: Chapter 3, or
  * Ivezić: Chapter 4
  
Lecture 06 (Tue 30 Jan) slides uploaded.  Covered:
  * Multi-parameter maximum likelihood, Fisher Matrix, and Cramer-Rao bound
  * Example: Fitting a Gaussian to data
  * Example: Fitting a Gaussian to data with measurement errors (Normal-Normal model)

**Week 3**  
Lecture 07 (Thu 01 Feb) slides uploaded.  Covered:
  * Demo about fitting Normal-Normal latent variable model with MLE
  * Rant about minimum chi^2 methods
  
Lecture 08 (Sat 03 Feb) slides uploaded. Covered:
  * Statistical Modeling wisdom
  * Ordinary Least Squares (OLS) and Generalised Least Squares (GLS) solutions to Linear Regression
  * Regression references: Ivezić, Ch 8; F & B Ch 7
  * Introduction to Latent Variable models with (y,x)-measurement errors and intrinsic dispersion

Lecture 09 (Tue 06 Feb) slides uploaded.  Covered:
  * Review Multivariate Normal Distribution.  See also Ivezić, Chapter 3
  * Probabilistic Generative / Forward Modeling of Data
  * example: Linear Regression with (y,x) measurement errors and intrinsic dispersion
  * reference: Kelly et al. 2007, The Astrophysical Journal, 665, 1489

**Week 4**  
Lecture 10 (Thu 08 Feb) slides uploaded. Also Covered:
  * Frequentist vs. Bayesian probability
  * confidence vs. credible intervals
  
Lecture 11 (Sat 10 Feb) slides uploaded.  Also covered:
  * Frequentist vs. Bayesian results for simple Gaussian
  * Null hypothesis testing, p-values
  * Likelihood principle, sufficient statistics, conjugate prior
  * Bayesian inference of simple Gaussian data with conjugate prior (Gelman BDA Ch 2-3)
  
Lecture 12 (Tue 13 Feb)...
  * Multi-parameter Bayesian analysis example (Gelman BDA, Sec 3.2-3.3),  
    Conjugate and "non-informative" prior
  * Posterior summaries & estimation
  * Monte Carlo Sampling
  * Importance Sampling

**Week 5**  
Bayesian Computation and Sampling Algorithms  

Lecture 13 (Thu 15 Feb).  Covered:
  * Kernel Density Estimation (F & B Ch 6, Ivezić, Sec 6.1.1)
  * Monte Carlo error
  * Importance Sampling
  * Case Study: Bayesian Estimates of the Milky Way Galaxy Mass
  * reference: Patel, Besla & Mandel. 2017, MNRAS, 468, 3428  
       MNRAS = Monthly Notices of the Royal Astronomical Society
       
Example Class 1 (Fri 16 Feb):
  * Solved Example Sheet 1, problems 1 & 2
  * Bootstrap Sampling (Ivezic, §4.5 & F&B §3.6.2)
       
Lecture 14 (Sat 17 Feb).  Covered:
  * Kernel Density Estimation (F & B Ch 6, Ivezić, Sec 6.1.1)
  * Review Bayesian Estimates of the Milky Way Galaxy Mass Case Study in more detail
  * Code Demonstration using Importance Sampling
  * Highest Posterior Density credible intervals
  
Lecture 15 (Tue 20 Feb). Covered:
  * Markov Chain Monte Carlo
  * Metropolis Algorithm
  * Example: Inferring a Gaussian mean, and code demo (metropolis1.m)

**Week 6**  
Lecture 16 (Thu 22 Feb). Covered:
  * Drawing multivariate Gaussian random variates (Cholesky factorisation)
  * Multi-dimensional Metropolis algorithm
  * Example & Code Demo: 2D metropolis, for Gaussian mean and variance
  * lecture_codes/metropolis1.m, metropolis2.m
  * Assessing convergence, Gelman-Rubin ratio
  * MCMC animations: http://chi-feng.github.io/mcmc-demo/

Lecture 17 (Sat 24 Feb). Covered:
  * Joint, Marginal, Conditional Properties of Multivariate Gaussians
  * Efficient Metropolis Proposal Rules
  * Metropolis-Hastings algorithms
  * Gibbs Sampling
  * Example & Code Demo: Gibbs Sampling a correlated bivariate Gaussian posterior
  * lecture_codes/gibbs_example.m

Lecture 18 (Tue 27 Feb). Covered:
  * Aside on structure of covariance matrices for Multivariate Gaussian dist'ns
  * Metropolis-within-Gibbs sampling
  * Began sketch of Markov Chain theoretical background

**Week 7**  
Lecture 19 (Thu 01 Mar). Covered:
  * Sketch of MCMC theory
  * irreducible, aperiodic, not transient chains with unique, stationary dist'ns
  * detailed balance and invariant dist'ns
  * proof that Metropolis-Hastings obeys detailed balance
  * Metropolis & Gibbs as special cases of M-H
  
Example Class 2 (Fri 02 Mar):
  * Solved Example Sheet 2, problems 1 & 2  
  * Properties of Multivariate Gaussian Distributions
  * see multivariate_gaussian_notes.pdf 
  * see also Appendix A of Rasmussen & Williams, "Gaussian Processes for Machine Learning"  
  http://www.gaussianprocess.org/gpml/chapters/
  
Lecture 20 (Sat 03 Mar). Covered:
  * MCMC in Practice
  * Choosing Metropolis proposal scale,
  * autocorrelation plots, thinning, multiple chains, Gelman-Rubin ratio
  
Lecture 21 (Tue 06 Mar):
  * Gaussian Processes in Astrophysics
  * Gravitational Lensing Time delay example
  * Fitting GP to data to infer functions
  
**Week 8**  
Lecture 22 (Thu 08 Mar):
  * Continue with GPs in Astrophysics  
  * Gravitational Lensing Example  
  * Optimising hyperparameters with marginal likelihood  
  
Lecture 23 (Sat 10 Mar):  
  * Jeffrey's priors revisited, results for Gaussian likelihood
  * Probabilistic Graphical Models & Hierarchical Bayes  
  * Case Study: Hierachical Bayesian Models for Type Ia Supernova Inference  
  * reference: Mandel et al. 2017, The Astrophysical Journal, 842, 93.  
  
**Example Class 3 (Mon 12 Mar, 15:30, MR 14)**  
  * Example Sheet 3: Doubly Lensed Quasar Time Delay estimation

Lecture 24 (Tue 13 Mar):  
  * Continue with PGMs and Hierarchical Bayes
  * Examples of hierarchical models / PGM in astronomy
  * Directed acyclic graphs, d-separation & conditional independence
  * Gibbs Sampling of hierarchical model
  * Supernova Cosmology example (Mandel et al. 2017)
  
**Example Class 4 (Thu 26 Apr, 13:00, MR 5)**
  * Went over Example Sheet 4, Problems 1, 4, 6
  
**Revision Class: Monday, 21 May 2018 at 10 am in MR 5**
  * Went over Example Sheet 4, Problems 5, 6
  
