Data Drive Model Predictive Control:
---------------------------------------------
------------------------------------------------------------------------------
Box Configuration
Inputs:
Environment:Site Outdoor Air Drybulb Temperature [C]
Environment:Site Outdoor Air Wetbulb Temperature [C]
Environment:Site Outdoor Air Relative Humidity [%]
Environment:Site Wind Speed [m/s]
Environment:Site Wind Direction [deg]
Environment:Site Horizontal Infrared Radiation Rate per Area [W/m2]
Environment:Site Diffuse Solar Radiation Rate per Area [W/m2]
Environment:Site Direct Solar Radiation Rate per Area [W/m2]
THERMAL ZONE: BOX:Zone Outdoor Air Wind Speed [m/s]

Outputs:
THERMAL ZONE: BOX:Zone Mean Air Temperature [C]


------------------------------------------------------------------------------
GSHP configuration
Inputs:
Environment:Site Outdoor Air Drybulb Temperature [C]
Environment:Site Outdoor Air Wetbulb Temperature [C]
Environment:Site Outdoor Air Relative Humidity [%]
Environment:Site Wind Speed [m/s]
Environment:Site Wind Direction [deg]
Environment:Site Horizontal Infrared Radiation Rate per Area [W/m2]
Environment:Site Diffuse Solar Radiation Rate per Area [W/m2]
Environment:Site Direct Solar Radiation Rate per Area [W/m2]
THERMAL ZONE: BOX:Zone Outdoor Air Wind Speed [m/s]
GSHPCLG:Heat Pump Electric Power [W]
GSHPCLG:Heat Pump Source Side Inlet Temperature [C]
GSHPHEATING:Heat Pump Electric Power [W]
GSHPHEATING:Heat Pump Source Side Inlet Temperature [C]


Outputs:
THERMAL ZONE: BOX:Zone Mean Air Temperature [C]

Data Driven MPC Approach. Online approach
steps:
    1) Predict room temperature at time t given current weather states
    2) Optimise for control at time t to reference room temperature
    3) Predict room temperature at time t+1 using temperature at time t
    4) Predict weather states for t+1 using temperature at t+1 (gotten from 3)
    4) Set for next evolution, temperature at t= temperature at t+1
      (prior(t)= posterior(t+1))
    
mathematically;    
y=room temperature
X=weather states
u=control for GSHP pump

input: u(t-1)= initial guess,X(t-1)(= Known), r for all t(= known), f1, f2,g (= Learned),
y(t-1)(=Infered from y(t-1)=f1(X(t-1))+e )
g=LSTM machine
f1=States to output (room temperature) machine (Pure weather conditions)
f2=Augmented states (with control inputs) to room temperature

set:
y(1)=y(t-1) 
X(1)=X(t-1) 
u(1)=u(t-1) 
Do t= 1: Horizon:
    y(t+1)=g(y(t))+n # Predict the future output given present output
    y(t)=f1(X(t))+e # Predict current output with current states
    ybig(t,:)=y(t)
    u(opt)=argmin||r(t)-f2(X(t);u(t),X(t))||+z # Optimise the control at time t
    ubig(t,:)=u(opt)
    Xbig(t,:)=X
    
    X(opt)=argmin||y(t+1)-f1(X(t)||+z # Optimise the state at time t+1
    set X(t)= X(t+1)=X(opt)
    set u(t)= u(t+1)=u(opt)
    
    
End Do

Getting Started:
---------------------------------
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
See deployment for notes on how to deploy the project on a live system.

-Run the script TRAINING.m to learn the machines of the LSTM, states to room temperature and states+contoller (GSHP pump) to temperature
-Optimise the controller input using the DD_MPC.m for Batch optimisation or DD_MPC_SEQUENTIAL_2 for sequential optimsation
Prerequisites:
-------------------------------
MATLAB 2018 upwards

Methods:
-------------------------------
Two methods are available for the optimisation problem;
- LBFGS
- Iterative Ensemble SMoother



Datasets
-----------------------------
Both datasets can be found i the Data directory
- The Weather dataset for 2 years with a daily frequency is called "Box.csv"
-The training dataset for the controller is found in the "GSHP.csv" FILE

Running the Numerical Experiment:
-Run the script TRAINING.m to learn the machines of the LSTM, states to room temperature and states+contoller (GSHP pump) to temperature
-Optimise the controller input using the DD_MPC.m for Batch optimisation or DD_MPC_SEQUENTIAL_2 for sequential optimsation

Dependencies
----------------------------


All libraries are included for your convenience.

Manuscript
-----------------------------
-

Author:
--------------------------------
Dr Clement Etienam- Research Officer-Machine Learning. Active Building Centre


Acknowledgments:
------------------------------


References:
----------------------------

[1] Luca Ambrogioni, Umut Güçlü, Marcel AJ van Gerven, and Eric Maris. The kernel mixture network: A non-parametric method for conditional density estimation 
of continuous random variables. arXiv preprint arXiv:1705.07111, 2017.

[2] Christopher M Bishop. Mixture density networks. 1994.

[3] Isobel C. Gormley and Sylvia Frühwirth-Schnatter. Mixtures of Experts Models. Chapman and Hall/CRC, 2019.

[4] R.B. Gramacy and H.K. Lee. Bayesian treed Gaussian process models with an application to computer modeling. Journal of the American Statistical Association, 103(483):1119–1130,
2008.

[5] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, Geoffrey E Hinton, et al. Adaptive
mixtures of local experts. Neural computation, 3(1):79–87, 1991.
2

[6] Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm.
Neural computation, 6(2):181–214, 1994.

[7] Trung Nguyen and Edwin Bonilla. Fast allocation of gaussian process experts. In International
Conference on Machine Learning, pages 145–153, 2014.

[8] Carl E Rasmussen and Zoubin Ghahramani. Infinite mixtures of gaussian process experts. In
Advances in neural information processing systems, pages 881–888, 2002.

[9] Tommaso Rigon and Daniele Durante. Tractable bayesian density regression via logit stickbreaking
priors. arXiv preprint arXiv:1701.02969, 2017.

[10] Volker Tresp. Mixtures of gaussian processes. In Advances in neural information processing
systems, pages 654–660, 2001.

[11] Lei Xu, Michael I Jordan, and Geoffrey E Hinton. An alternative model for mixtures of experts.

[12] Rasmussen, Carl Edward and Nickisch, Hannes. Gaussian processes for machine learning (gpml) toolbox. The
Journal of Machine Learning Research, 11:3011–3015, 2010

[13] David E. Bernholdt, Mark R. Cianciosa, David L. Green, Jin M. Park, Kody J. H. Law, and
Clement Etienam. Cluster, classify, regress: A general method for learning discontinuous functions. Foundations of Data Science, 
1(2639-8001-2019-4-491):491, 2019.

[14] Clement Etienam, Kody Law, Sara Wade. Ultra-fast Deep Mixtures of Gaussian Process Experts. arXiv preprint arXiv:2006.13309, 2020.