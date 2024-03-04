import numpy as np
import time
import pylsl
from typing import List


from HIL.optimization.BO import BayesianOptimization
from HIL.optimization.MOBO import MultiObjectiveBayesianOptimization
from HIL.optimization.extract_cost import ExtractCost


class HIL:
    """Main HIL optimization
    This program will extract cost from the pylsl stream.
    Run the optimization.
    Check if the optimization is done.
    """

    def __init__(self, args: dict) -> None:
        """cost_name: name of the cost function."""
        self.n = int(0)  # number of optimization
        self.x = np.array([])  # input parameter for the exoskeleton
        # self.y = np.array([]) # cost function
        self.args = args
            # normalization
        if self.args["Optimization"]["normalize"]:
            self.NORMALIZATION = True
        else:
            self.NORMALIZATION = False
        
        if self.args["Optimization"]["MultiObjective"] == 1:
            self.MULTI_OBJECTIVE = True
        else:
            self.MULTI_OBJECTIVE = False


        # start the
        self.start_time = 0

        self._outlet_cost()

        self._reset_data_collection()

        # start optimization
        self._start_optimization(self.args["Optimization"])

        # start cost function
        self._start_cost(self.args["Cost"])

        # self.warm_up
        self.warm_up = True

        # start optimization
        self.OPTIMIZATION = False


        # The ones which are done.
        self.x_opt = np.array([])
        self.y_opt = np.array([])


    def _normalize_x(self, x: np.ndarray) -> np.ndarray:
        """Normalize x based on the range of the parameter

        Args:
            x (np.ndarray): x - parameters ( optimization parameters )

        Returns:
            np.ndarray: Normalized X parameters ( optimization parameters )
        """
        x = np.array(x).reshape(self.n, self.args["Optimization"]["n_parms"])
        range_x = np.array(self.args["Optimization"]["range"]).reshape(2, self.args["Optimization"]['n_parms'])
        x = (x - range_x[0,:]) / (range_x[1, :] - range_x[0, :])
        return x

    def _denormalize_x(self, x: np.ndarray) -> np.ndarray:
        """Denormalize x based on the range of the parameter, This is to send the parameters to exokseleton.

        Args:
            x (np.ndarray): x - parameters ( optimization parameters )

        Returns:
            np.ndarray: Denormalized X parameters ( optimization parameters )
        """
        x = np.array(x).reshape(-1, self.args["Optimization"]["n_parms"])
        range_x = np.array(self.args["Optimization"]["range"]).reshape(2, self.args["Optimization"]['n_parms'])
        x = x * (range_x[1, : ] - range_x[0, :]) + range_x[0, :]
        return x

    def _mean_normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Mean normalize y based on the range of the parameter

        Args:
            y (np.ndarray): y - cost function

        Returns:
            np.ndarray: Normalized y - cost function
        """
        y = np.array(y)
        y = (y - np.mean(y)) / np.std(y)
        return y

    def _mean_normalize_y_multi(self, y: np.ndarray) -> np.ndarray:
        """Mean normalize y based on the range of the parameter

        Args:
            y (np.ndarray): y - cost function

        Returns:
            np.ndarray: Normalized y - cost function
        """
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        y_normalized = (y - y_mean) / y_std

        return y_normalized

    def _outlet_cost(self) -> None:
        """Create an outlet function to send when the optimization has changed"""
        if self.MULTI_OBJECTIVE==True:
            self.chan_count = 3
        else:
            self.chan_count = 2

        info = pylsl.StreamInfo(
            name="Change_parm",
            type="Marker",
            channel_count=self.chan_count,   # Channel count is 2, if using single objective, and 3 if using multi objective 
            source_id="12345",
        )
        self.outlet = pylsl.StreamOutlet(info)
    # channel_count = self.args["Optimization"]["n_parms"] + 1,
        
    def _reset_data_collection(self) -> None:
        """Reset data collection and restart the clocks"""
        if self.MULTI_OBJECTIVE:
            self.store_cost_data = [[] for i in range(0,len(self.args['Multi_Objective_Cost']))]  #list of lists
        else:
            self.store_cost_data = []
        self.cost_time = 0
        self.start_time = 0

        # cost_time represents the timestamp associated with the latest cost data received from the Lab Streaming Layer (LSL) stream.
        # During the cost extraction process (_get_cost method), the timestamp of the most recent cost data is stored in cost_time.

        # start_time represents the timestamp associated with the beginning of the data collection for a specific phase (either exploration or optimization).
        # It is set when the first piece of cost data is received during a particular phase.
        # if len(self.store_cost_data) == 1:
            # self.start_time = time_stamp
        # This timestamp is used to calculate the elapsed time during the current phase (time: {self.cost_time - self.start_time}).
    
    def _start_optimization(self, args: dict) -> None:
        """Start the optimization function, this will start the BO module/MOBO module (if multi-objective is selected)

        Args:
            args (dict): Optimization args
        """
        print(args["range"][0], args["range"][1])
        print(np.array(list(args["range"])))
        if self.MULTI_OBJECTIVE:
            self.MOBO = MultiObjectiveBayesianOptimization()
        else:
            self.BO = BayesianOptimization(
                n_parms=args["n_parms"],
                range=np.array(list(args["range"])),
                model_save_path=args["model_save_path"],
                acq = args["acquisition"]
            )

    def _start_cost(self, args: dict) -> None:
        """Start the cost extraction module

        Args:
            args (dict): Cost args
        """
        self.cost_time = 0
        if self.MULTI_OBJECTIVE:
            # this is for multi objective
            self.cost = [] # list of objects of ExtractCost class (length equal to number of objectives)
            
            # self.args['Multi_Objective_Cost'] is a dictionary of dictionaries (Cost1, Cost2), each with Name and n_samples as key-value pairs
            # name represents the key (Cost1 or Cost2)
            # cost_config represents the value, which is a dictionary containing configuration information for the corresponding cost.
            # (Name: ECG_processed, n_samples: 5...) or (Name: ECG_complexity, n_samples: 5...)
            
            for name, cost_config in self.args['Multi_Objective_Cost'].items():    
                self.cost.append(ExtractCost(
            cost_name=cost_config["Name"], number_samples=cost_config["n_samples"]
        ))
        else:
            # This is for single objective
            self.cost = ExtractCost(
                cost_name=args["Name"], number_samples=args["n_samples"]
            )

    def start(self):
        print('start')
        if self.MULTI_OBJECTIVE:
            if self.n == 0:
                print(f"############################################################")
                print(f"############## Starting the optimization ###################")
                print(f"############## Using cost function (multi-objective) #######")
                print(f"############################################################")
                self._generate_initial_parameters()
                self.outlet.push_sample(self.x[0,:].tolist() + [0, 0])  # cost has two objectives
            # start the optimization loop.
            while self.n < self.args["Optimization"]["n_steps"]:
                # Still in exploration
                if self.n < self.args["Optimization"]["n_exploration"]:
                    print(
                        f"In the exploration step {self.n}, parameter {self.x[self.n]}, len_cost {len(self.store_cost_data[0])}"
                    )
    
                    if self.n == 0 and self.warm_up:
                        input(f"Please give 2 min of warmup and hit any key to continue \n")
                        self.warm_up = False
    
                    self._get_cost()    # here, self.start_time is set to time_stamp (time stamp of the cost function from the LSL)
                    self.outlet.push_sample(self.x[self.n,:].tolist() + [0, 0])   # cost has two objectives
                    if (self.cost_time - self.start_time) > self.args["Cost"][
                        "time"
                    ] and len(
                        self.store_cost_data[0]
                    ) > 5:  # 30 for 120
    
                        # Calculate the mean cost
                        # Extract the last 5 elements for each objective
                        last_5_elements = [obj[-5:] for obj in self.store_cost_data]
    
                        # Calculate the mean for each objective
                        mean_costs = [np.mean(obj) for obj in last_5_elements] # mean_costs = [__, __]
    
                        print(f" cost is {mean_costs}")
                        # add reset time function 
                        # self.start_time=0 or time_stamp?
                        out = input("Press Y to record the data: N to remove it:")
                        print('self.start_time ' + str(self.start_time) + 'self.cost_time ' + str(self.cost_time))
                        if out == "N":
                            self._reset_data_collection()
                            print("#########################")
                            print("########### recollecting #######")
                            print("#########################")
                        else:
                            if len(self.x_opt) < 1:
                                self.x_opt = np.array([self.x[self.n]])
                            else:
                                self.x_opt = np.concatenate(
                                    (self.x_opt, np.array([self.x[self.n]]))
                                )
                            
                            # Extract the last 5 elements for each objective
                            last_5_elements = [obj[-5:] for obj in self.store_cost_data]
    
                            # Calculate the mean cost for each objective
                            mean_costs = [np.mean(obj) for obj in last_5_elements]
    
                            if len(self.y_opt) < 1:
                                self.y_opt = np.array([mean_costs])
                            else:
                                self.y_opt = np.concatenate(
                                    (self.y_opt, np.array([mean_costs]))
                                )
    
                            print(
                                f"recording cost function {self.y_opt[-1]}, for the parameter {self.x_opt[-1]}"
                            )
    
                            self.outlet.push_sample(
                                self.x_opt[-1].tolist() + self.y_opt[-1].flatten().tolist()         # cost has two objectives
                            )   
                            # The push_sample method expects a list for the sample to push into the outlet. 
                            # If y_opt[-1] is a 2D array and you want to push it as a single sample, you can flatten it to a 1D list before pushing. 
                            # y_opt[-1] is flattened to a 1D list and concatenated with x_opt[-1].tolist()

                            self._reset_data_collection()
                            self.n += 1
                            input("Enter to Continue")
    
                # Exploration is done and starting the optimization
                elif (
                    self.n == self.args["Optimization"]["n_exploration"]
                    and not self.OPTIMIZATION
                ):
    
                    # Extract the last 5 elements for each objective
                    last_5_elements = [obj[-5:] for obj in self.store_cost_data]
    
                    # Calculate the mean cost for each objective
                    mean_costs = [np.mean(obj) for obj in last_5_elements]
                    
                    print(f" cost is {mean_costs}")
                    
                    out = input("Press Y to record the data: N to remove it:")
                    print('self.start_time ' + str(self.start_time) + 'self.cost_time ' + str(self.cost_time))
                    if out == "N":
                        self._reset_data_collection()
                        print("################################")
                        print("########### recollecting #######")
                        print("################################")
                    else:
                        print(f"starting the optimization.")
                        if self.NORMALIZATION:
                            norm_x = self._normalize_x(self.x_opt)
                            norm_y = self._mean_normalize_y_multi(self.y_opt)
                            new_parameter = self.MOBO.generate_next_candidate(norm_x,norm_y)
                            print(f"Next parameter without norm is {new_parameter}")
                            new_parameter = self._denormalize_x(new_parameter)
    
                        else:
                            new_parameter = self.MOBO.generate_next_candidate(
                                self.x_opt,
                                self.y_opt,
                            )
                        
                        print(f"Next parameter is {new_parameter}")
                        self.outlet.push_sample(self.x_opt[-1].tolist() + self.y_opt[-1].flatten().tolist())     # cost has two objectives
    
                        # TODO Need to save the parameters and data for each iteration,
                        self.x = np.concatenate(
                            (
                                self.x,
                                new_parameter.reshape(
                                    1, self.args["Optimization"]["n_parms"]
                                ),
                            ),
                            axis=0,
                        )
                        self.OPTIMIZATION = True

                else:
                    print(f"In the optimization loop {self.n}, parameter {self.x[self.n]}")
                    self._get_cost()
                    self.outlet.push_sample(self.x[self.n,:].tolist() + [0, 0])    # cost has two objectives
                    if (self.cost_time - self.start_time) > self.args["Cost"]["time"]:
                        out = input("Press Y to record the data: N to remove it:")
                        print('self.start_time ' + str(self.start_time) + 'self.cost_time ' + str(self.cost_time))
                        if out == "N":
                            self._reset_data_collection()
                            print("################################")
                            print("########### recollecting #######")
                            print("################################")
                        else:
                            self.x_opt = np.concatenate(
                                (self.x_opt, np.array([self.x[self.n]]))
                            )
                            
                            # Extract the last 5 elements for each objective
                            last_5_elements = [obj[-5:] for obj in self.store_cost_data]
    
                            # Calculate the mean cost for each objective
                            mean_costs = [np.mean(obj) for obj in last_5_elements]
                            
                            self.y_opt = np.concatenate((self.y_opt, np.array([mean_costs])))
                            self.n += 1
                            print(
                                f"recording cost function {self.y_opt[-1]}, for the parameter {self.x_opt[-1]}"
                            )
                            if self.NORMALIZATION:
                                norm_x = self._normalize_x(self.x_opt)
                                norm_y = self._mean_normalize_y_multi(self.y_opt)
                                new_parameter = self.MOBO.generate_next_candidate(norm_x, norm_y)
                                new_parameter = self._denormalize_x(new_parameter)
    
                            else:
                                new_parameter = self.BO.run(
                                    self.x_opt,
                                    self.y_opt,
                                )
                            print(f"Next parameter is {new_parameter}")
                            # TODO Need to save the parameters and data for each iteration
                            self.x = np.concatenate(
                                (
                                    self.x,
                                    new_parameter.reshape(
                                        1, self.args["Optimization"]["n_parms"]
                                    ),
                                ),
                                axis=0,
                            )
                            # self.outlet.push_sample([self.x_opt[self.n,:].tolist(), self.y_opt[-1].tolist()])
                            self._reset_data_collection()
                            input("Enter to contiue")
                time.sleep(1)

        else:
            if self.n == 0:
                print(f"############################################################")
                print(f"############## Starting the optimization ###################")
                print(f"############## Using cost function {self.cost.cost_name} ###")
                print(f"############################################################")
                self._generate_initial_parameters()
                self.outlet.push_sample(self.x[0,:].tolist() + [0])
            # start the optimization loop.
            while self.n < self.args["Optimization"]["n_steps"]:
                print('n = ', self.n)
                # Still in exploration
                if self.n < self.args["Optimization"]["n_exploration"]:
                    print(
                        f"In the exploration step {self.n}, parameter {self.x[self.n]}, len_cost {len(self.store_cost_data)}"
                    )
    
                    if self.n == 0 and self.warm_up:
                        input(f"Please give 2 min of warmup and hit any key to continue \n")
                        self.warm_up = False
    
                    self._get_cost()
                    print('self.store_cost_data ', self.store_cost_data)
                    print('get cost done')
                    self.outlet.push_sample(self.x[self.n,:].tolist() + [0])
                    if (self.cost_time - self.start_time) > self.args["Cost"][
                        "time"
                    ] and len(
                        self.store_cost_data
                    ) > 5:  # 30 for 120
                        print(f" cost is {np.mean(self.store_cost_data[-5:])}")
                        out = input("Press Y to record the data: N to remove it:")
                        print('self.start_time ' + str(self.start_time) + 'self.cost_time ' + str(self.cost_time))
                        if out == "N":
                            self._reset_data_collection()
                            print("#########################")
                            print("########### recollecting #######")
                            print("#########################")
                        else:
                            if len(self.x_opt) < 1:
                                self.x_opt = np.array([self.x[self.n]])
                            else:
                                self.x_opt = np.concatenate(
                                    (self.x_opt, np.array([self.x[self.n]]))
                                )
                            mean_cost = np.mean(self.store_cost_data[-5:])
    
                            if len(self.y_opt) < 1:
                                self.y_opt = np.array([mean_cost])
                            else:
                                self.y_opt = np.concatenate(
                                    (self.y_opt, np.array([mean_cost]))
                                )
    
                            print(
                                f"recording cost function {self.y_opt[-1]}, for the parameter {self.x_opt[-1]}"
                            )
    
                            self.outlet.push_sample(
                                self.x_opt[-1].tolist() + [self.y_opt[-1]]
                            )
                            self._reset_data_collection()
                            self.n += 1
                            input("Enter to Continue")
    
                # Exploration is done and starting the optimization
                elif (
                    self.n == self.args["Optimization"]["n_exploration"]
                    and not self.OPTIMIZATION
                ):
                    print(f" cost is {np.mean(self.store_cost_data[-5:])}")
                    out = input("Press Y to record the data: N to remove it:")
                    print('self.start_time ' + str(self.start_time) + 'self.cost_time ' + str(self.cost_time))
                    if out == "N":
                        self._reset_data_collection()
                        print("################################")
                        print("########### recollecting #######")
                        print("################################")
                    else:
                        print(f"starting the optimization.")
                        if self.NORMALIZATION:
                            norm_x = self._normalize_x(self.x_opt)
                            norm_y = self._mean_normalize_y(self.y_opt)
                            new_parameter = self.BO.run(norm_x.reshape(self.n, -1), norm_y.reshape(self.n, -1))
                            print(f"Next parameter without norm is {new_parameter}")
                            new_parameter = self._denormalize_x(new_parameter)
    
                        else:
                            new_parameter = self.BO.run(
                                self.x_opt.reshape(self.n, -1),
                                self.y_opt.reshape(self.n, -1),
                            )
                        
                        print(f"Next parameter is {new_parameter}")
                        self.outlet.push_sample(self.x_opt[-1].tolist() + [self.y_opt[-1]])
    
                        # TODO Need to save the parameters and data for each iteration,
                        self.x = np.concatenate(
                            (
                                self.x,
                                new_parameter.reshape(
                                    1, self.args["Optimization"]["n_parms"]
                                ),
                            ),
                            axis=0,
                        )
                        self.OPTIMIZATION = True
    
                else:
                    print(f"In the optimization loop {self.n}, parameter {self.x[self.n]}")
                    self._get_cost()
                    self.outlet.push_sample(self.x[self.n,:].tolist() + [0])
                    if (self.cost_time - self.start_time) > self.args["Cost"]["time"]:
                        out = input("Press Y to record the data: N to remove it:")
                        print('self.start_time ' + str(self.start_time) + 'self.cost_time ' + str(self.cost_time))
                        if out == "N":
                            self._reset_data_collection()
                            print("################################")
                            print("########### recollecting #######")
                            print("################################")
                        else:
                            self.x_opt = np.concatenate(
                                (self.x_opt, np.array([self.x[self.n]]))
                            )
                            mean_cost = np.mean(self.store_cost_data[-5:])
                            self.y_opt = np.concatenate((self.y_opt, np.array([mean_cost])))
                            self.n += 1
                            print(
                                f"recording cost function {self.y_opt[-1]}, for the parameter {self.x_opt[-1]}"
                            )
                            if self.NORMALIZATION:
                                norm_x = self._normalize_x(self.x_opt)
                                norm_y = self._mean_normalize_y(self.y_opt)
                                new_parameter = self.BO.run(norm_x.reshape(self.n, -1), norm_y.reshape(self.n, -1))
                                new_parameter = self._denormalize_x(new_parameter)
    
                            else:
                                new_parameter = self.BO.run(
                                    self.x_opt.reshape(self.n, -1),
                                    self.y_opt.reshape(self.n, -1),
                                )
                            print(f"Next parameter is {new_parameter}")
                            # TODO Need to save the parameters and data for each iteration
                            self.x = np.concatenate(
                                (
                                    self.x,
                                    new_parameter.reshape(
                                        1, self.args["Optimization"]["n_parms"]
                                    ),
                                ),
                                axis=0,
                            )
                            # self.outlet.push_sample([self.x_opt[self.n,:].tolist(), self.y_opt[-1].tolist()])
                            self._reset_data_collection()
                            input("Enter to contiue")
                time.sleep(1)       
        
            

    def _generate_initial_parameters(self) -> None:
        opt_args = self.args["Optimization"]
        range_ = np.array(list(opt_args["range"]))
        # generate the initial parameters in the range of the parameter and in the shape of exploration x shape
        self.x = np.random.uniform(
            range_[0], range_[1], size=(opt_args["n_exploration"], opt_args["n_parms"])
        )
        # self.x[0]=35.0
        # self.x[1]=75.0
        # self.x[2]=10.0
        print(f"###### start functions are {self.x} ######")

    def _get_cost(self) -> None:
        print('calling _get_cost')
        """This function extracts cost from pylsl, need to be called all the time."""
        if self.MULTI_OBJECTIVE:
            for i, cost in enumerate(self.cost):
                data, time_stamp = cost.extract_data()

                if time_stamp is not None:
                    # changing maximization to minimization.
                    data = data[-1] * -1
                    self.cost_time = time_stamp
                    self.store_cost_data[i].append(data)
                    if len(self.store_cost_data[i]) == 1:
                        self.start_time = time_stamp         
                    print(
                        f"got cost {self.store_cost_data[i][-1]}, parameter {self.x[self.n]}, time: {self.cost_time - self.start_time}"
                    )


        else:
            data, time_stamp = self.cost.extract_data()

            if time_stamp is not None:
                # changing maximization to minimization.
                data = data[-1] * -1
                self.cost_time = time_stamp
                self.store_cost_data.append(data)
                if len(self.store_cost_data) == 1:
                    print('updating start time to ', time_stamp)
                    self.start_time = time_stamp
                print(
                    f"got cost {self.store_cost_data[-1]}, parameter {self.x[self.n]}, time: {self.cost_time - self.start_time}"
                )
