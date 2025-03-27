import numpy as np
# import jax.numpy as jnp
# import jax.random as random

class MyPayoff:
    def __init__(self,mu,sigma,cov_matrix,r_f):
        self.S0 = np.ones(np.shape(sigma)) ## don't care about S0 since we normalise everything, this is a relic
        self.mu = mu
        self.sigma = sigma
        self.cov_matrix = cov_matrix
        self.r_f = r_f
        self.r_f_d = (1 + self.r_f) ** (1/252) - 1

    def _simulate_paths(self,T,N,n_iters):
        """
        GBM simulation
        output shape: num_paths, num_iters, num_udls
        """
        dt = T / N
        num_assets = len(self.S0)
    
        ## pre-computed for efficiency
        p1 = np.exp((self.mu - 0.5 * self.sigma**2) * dt)
        sigmaSqrtDt = self.sigma * np.sqrt(dt)
    
        paths = np.zeros([n_iters,N,num_assets])
        for i in range(n_iters):
            newPrice = self.S0     
            p2 = np.random.multivariate_normal(np.zeros(num_assets), self.cov_matrix, N) * sigmaSqrtDt

            for t in range(1, N):
                
                newPrice = newPrice * p1 * np.exp(p2[t])
                paths[i][t] = newPrice / self.S0
        
        self.paths = paths

    def _calculate_product_price(self,
                  observations, # date:level
                  couponTrue,
                  couponFalse,
                  basket_func=min,
                  options=[]
                 ):

        (n_iters, N,num_udls) = np.shape(self.paths)

        payments = {i:0 for i in observations.keys()}
        autocall_probabilities = {i:0 for i in observations.keys()}
        autocall_probabilities["end"] = 0

        path_end = np.zeros([n_iters])

        option_prices = [0]*len(options)

        for this_iter in range(n_iters):
            for t in range(N):
                if t in observations.keys():
                    if basket_func(self.paths[this_iter,t]) > observations[t]:
                        payments[t] += couponTrue[t]
                        autocall_probabilities[t] += 1
                        path_end[this_iter] = -1000
                        break
                    elif basket_func(self.paths[this_iter,t]) < observations[t]:
                        payments[t] += couponTrue[t]

                if t == N-1:
                    autocall_probabilities["end"] += 1
                    path_end[this_iter] = basket_func(self.paths[this_iter,t])
                    for i in range(len(options)):
                        option_prices[i] += options[i]._single_path_price(self.paths[this_iter])
                
        
        print("Options")
        option_prices = [i / n_iters for i in option_prices]
        print(f"\tUniscounted options: {option_prices}")
        option_prices = [i * 1/(1+self.r_f_d)**N for i in option_prices]
        
        print(f"\tDiscounted options: {option_prices}")
        print(f"\tOptions (pv): {sum(option_prices)*100:.4f} %")

        self.path_end = path_end
        for i in autocall_probabilities.keys():
            autocall_probabilities[i] /= n_iters

        pv_coupons = 0
        prev = 1
        for this_key in sorted(payments.keys()):
            temp = 1/(1+self.r_f_d)**this_key * couponTrue[this_key] * autocall_probabilities[this_key]
            prev = (1 - autocall_probabilities[this_key]) * prev
            temp += 1/(1+self.r_f_d)**this_key * couponFalse[this_key] * prev
            pv_coupons += temp

        duration = 0
        temp = None
        for this_key in autocall_probabilities:
            
            if this_key == "end":
                duration += autocall_probabilities[this_key] * temp
            else:
                duration += autocall_probabilities[this_key] * this_key
            
            temp=this_key
        
        self.duration = duration/252
        print("\nAutocall")
        print(f"\tDuration: {self.duration}")

        self.autocall_probabilities = autocall_probabilities
        self.pv_coupons = pv_coupons

        print(f"\tAutocall coupons (pv): {pv_coupons*100:.4f} %")
        
        print()
        print(f"\t{'Tenor':<8} {'Prob':<8}")
        for i,j in autocall_probabilities.items():
            print(f"\t{i:<8} {j:<8}")

class Option:
    def __init__(self,
                 
                 option_type,
                 K,
                 gearing,

                 barrier_type=None,
                 barrier_level=None,
                 
                 way=None,
                 knocks=None,

                 floor=None,

                 basket_func = lambda x : np.min(x,axis=1),
                 

                 ):

        """
        NOTE: basket_func must take an array of [T/dt,num_udls] and return one of [T/dt,]
        """
        
        if option_type not in ["Call", "Put"]:
            raise ValueError(f"Option type not supported: {option_type}")
        self.option_type = option_type
        self.K = K
        self.gearing = gearing


        if barrier_type not in ["European", "American",None]:
            raise ValueError(f"Barrier type not supported: {barrier_type}")
        self.barrier_level = barrier_level
        self.barrier_type = barrier_type
        self.way = way
        self.knocks = knocks

        self.floor = floor

        self.basket_func = basket_func

    def _single_path_price(self,path):
        path = self.basket_func(path)
        # if not path_end == path[-1]:
        #     print(path[-1],path_end)
        

        if self.option_type == "Call":
            payoff = lambda p: 0 if p is None else max(p-self.K,0)
        elif self.option_type == "Put":
            payoff = lambda p: 0 if p is None else  max(self.K-p,0)
        else:
            raise AssertionError("Not sure how we got here")

        
        final_price = path[-1]
        if final_price < 0:
            print("Why is final price less than zero??")
            return 0
        observed_level_for_barrier = path[-1]

        price = payoff(final_price)

        if self.barrier_type:
            if self.barrier_type == "European":
                observed_level_for_barrier = final_price
            elif self.barrier_type == "American":
                if self.way == "Down":
                    observed_level_for_barrier = np.min(path)
                elif self.way == "Up":
                    observed_level_for_barrier = np.max(path)

            if self.knocks == "Knockout":
                if self.way == "Down":
                    price = 0 if observed_level_for_barrier < self.barrier_level else final_price
                elif self.way == "Up":
                    price = 0 if observed_level_for_barrier > self.barrier_level else final_price

            elif self.knocks == "Knockin":
                if self.way == "Down":
                    price = 0 if observed_level_for_barrier > self.barrier_level else final_price
                elif self.way == "Up":
                    price = 0 if observed_level_for_barrier < self.barrier_level else final_price
        
        price *= self.gearing

        return price        
            


if __name__ == "__main__":

    r_f = 4/100
    mu = np.array([0.04,0.03])  # Expected returns of the assets
    sigma = np.array([0.4809,0.2467])  # Volatilities of the assets
    cov_matrix = np.array(
        [
            [sigma[0]**2, sigma[0]*sigma[1]*0.4349], 
            [sigma[0]*sigma[1]*0.4349, sigma[1]**2]
        ]
    )  # Covariance matrix of the assets


    # S0 = np.array([113.76])  # Initial prices of two assets
    # mu = np.array([0.03])  # Expected returns of the assets
    # sigma = np.array([0.4778])  # Volatilities of the assets
    # cov_matrix = np.array(
    #     [
    #         [sigma[0]**2], 
    #         # [sigma[0]*sigma[1]*0.4349, sigma[1]**2]
    #     ]
    # )  # Covariance matrix of the assets

    T = 2  # Time to maturity in years
    N = 252*T  # Number of time steps (e.g. daily steps for one year)
    n_iters = 1000 # Number of simulation iterations

    # S0 = np.array([113.1,284.05])  # Initial prices of two assets
    # mu = np.array([0.03,0.03])  # Expected returns of the assets
    # sigma = np.array([0.4756,0.6290])  # Volatilities of the assets
    # cov_matrix = np.array(
    #     [
    #         [sigma[0]**2, sigma[0]*sigma[1]*0.6535], 
    #         [sigma[0]*sigma[1]*0.6535, sigma[1]**2]
    #     ]
    # )  # Covariance matrix of the assets

    ## product_description
    # observations = {63: 1, 126: 1, 189: 1, 252: 1}
    # observations = {63: 3, 126: 3, 189: 3, 252: 3}
    # couponTrue = {63: 0.05, 126: 0.05, 189: 0.05, 252: 0.05}
    # couponFalse = {63: 0.0, 126: 0.0, 189: 0.0, 252: 0.0}

    observations = {63.0: 1, 126.0: 1, 189.0: 1, 252.0: 1, 315.0: 1, 378.0: 1, 441.0: 1, 504.0: 1}
    # observations = {63.0: 1.2, 126.0: 1.2, 189.0: 1.2, 252.0: 1.2, 315.0: 1.2, 378.0: 1.2, 441.0: 1.2, 504.0: 1.2}
    # observations = {63.0: 3, 126.0: 3, 189.0: 3, 252.0: 3, 315.0: 3, 378.0: 3, 441.0: 3, 504.0: 3}
    couponTrue = {63.0: 0.02,
                    126.0: 0.02,
                    189.0: 0.02,
                    252.0: 0.02,
                    315.0: 0.02,
                    378.0: 0.02,
                    441.0: 0.02,
                    504.0: 0.02}
    couponFalse = {63.0: 0.01,
                    126.0: 0.01,
                    189.0: 0.01,
                    252.0: 0.01,
                    315.0: 0.01,
                    378.0: 0.01,
                    441.0: 0.01,
                    504.0: 0.01}


    a = MyPayoff(mu,sigma,cov_matrix,r_f)
    a._simulate_paths(T,N,n_iters)
    a._calculate_product_price(
        observations = observations,
        couponTrue = couponTrue,
        couponFalse = couponFalse,
        options = [
            Option(
                "Put",
                1,
                1,
                barrier_type="American",
                barrier_level=0.8,
                
                way="Down",
                knocks="Knockin",
            ),
            Option(
                "Put",
                1,
                -1,
                barrier_type="American",
                barrier_level=0.8,
                
                way="Down",
                knocks="Knockin",
            ),
        ]
    )
    

    