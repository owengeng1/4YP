'''
vect_classes.py
By: Owen Geng
Contains the relevant object information used by the main testbench.
'''

import numpy as np
import random
from math_operations import *
#import linalg



class World:
    #Initialiser, takes in the problem space dimensions and a predefined size, formatted as [[d1min, d1max], [d2min, d2max], [d3min, d3max] ...]
    def __init__(self, dim = None, sizearr = None):
        if dim is None and sizearr is None:
            self.dim = 3 #Defaults to 3D
            print("No information provided, setting dimensions to 3 and size array to default.")
        
        if dim is None:
            self.dim = np.shape(sizearr)[0] #Sets dimensions to the number of rows in sizearr
            print(f"{dim} dimensions detected in inputted size array.")
            self.sizearr = sizearr
        
        if sizearr is None:
            self.dim = dim
            self.sizearr = np.array([-100, 100])
            #print("No size array provided, setting to default.") #Default will be from -100 to 100 in every direction. Might need to change this to save memory at some point.
            for i in range (dim-1):
                self.sizearr = np.vstack((self.sizearr, [-100, 100]))
            
        
        self.vect_list = []
    
    def add_vector(self, vector):
        assert isinstance(vector, Vector), "Vector added must be of class Vector."
        self.vect_list.append(vector)

    def delete_vector(self, vect_ind):
        self.vect_list.pop(vect_ind)
    
    def generate(self, dim1, start = None, dir = None):
        if start is None: #If no starting point defined
            dir1 = np.zeros(self.dim)
            for ind in range(self.dim): #Random starting point within space defined by sizearr.
                dir1[ind] = random.uniform(self.sizearr[ind][0], self.sizearr[ind][1])
        else:
            dir1 = start
        
        if dir is None: #If no direction defined
            kstore = np.zeros(self.dim)
            for kn in range(dim1):
                for ind in range(self.dim): #Random directions.
                    kstore[ind] = random.gauss(0, 1)
                dir1 = np.vstack((dir1, kstore))
        else:
            if np.shape(dir)[0] < dim1:
                kstore = np.zeros(self.dim)
                for kn in range (dim1 - np.shape(dir)[0]): #If not all direction dimensions are defined, fills in the rest.
                    for ind in range(self.dim): #Random directions.
                        kstore[ind] = random.gauss(0, 1)
                dir = np.vstack((dir, kstore))
            
            dir1 = np.vstack((dir1, dir))
        
        vect1 = Vector(dim1, dir1)
        vect1.normalise_dir()
        return vect1
    
    def generate_pair(self, dim1, dim2, intersect = False):
        vect1 = self.generate(dim1)
        if intersect == False:
            vect2 = self.generate(dim2)
        else:
            raise(NotImplementedError)
        
        self.vect_list.append(vect1)
        self.vect_list.append(vect2)
    
    def generate_angle(self, ref_line, angle, start = None): #Generate a vector that is a certain angle from a predefined line. Currently only works for 1D.
        assert isinstance(ref_line, Vector), "generate_angle ref_line must be of class Vector"
        assert ref_line.dim == 1, "generate_angle currently only implemented for 1D case"
        #Generate a random point
        dir1 = np.zeros(self.dim)
        for ind in range(self.dim):
            dir1[ind] = random.uniform(self.sizearr[ind][0], self.sizearr[ind][1])
        rand_point = np.array(dir1)
        init_dir = ref_line.dirarr[1]
        #Find point's projection in hyperplane orthogonal to ref_line
        #OA . OC = |OA| |OC| cos(theta), proj(C) = OC - OB where OB = magnitude of OC in direction OA, i.e, |OC| cos(theta) or OA.OC/|OA|
        dot = np.dot(init_dir, rand_point)
        dot = dot/(calculate_magnitude(init_dir))
        proj = np.add(-dot*(init_dir), rand_point)
        proj = proj/calculate_magnitude(proj)
        
        #Using projected point to generate a new line at a predefined angle to init_dir
        final_dir = np.cos(angle)*init_dir + np.sin(angle)*proj
        final_dir = final_dir/(calculate_magnitude(final_dir))

        #Create new Vector object with a random starting point/predefined starting point.
        #Generate a random point
        if start == None:
            start = np.zeros(self.dim)
            for ind in range(self.dim):
                start[ind] = random.uniform(self.sizearr[ind][0], self.sizearr[ind][1])
        else:
            assert isinstance(start, np.ndarray), "Starting point must be an np array"
        
        return Vector(1, np.vstack((start, final_dir)))
        

    def calc_l2(self, p1, p2):
        assert np.size(p1) == np.size(p2), "Two points do not have the same length."

        return np.sum((np.subtract(p1, p2)**2))
    

    def step_descent(self, vect1, vect2):
        #Initialise variables
        K1 = np.zeros(vect1.dim) #Initial guess - everything is 0 along all constants
        K2 = np.zeros(vect2.dim)
        converged = False #While check
        iteration_counter = 0
        while converged == False:
            dist_minimum = self.calc_l2(vect1.calculate_point(K1), vect2.calculate_point(K2))
            for i1, k1n in enumerate(K1):
                K1temp = np.copy(K1) #Find the probing points
                K1temp[i1] = k1n + 0.1
                p11 = vect1.calculate_point(K1temp)
                K1temp[i1] = k1n - 0.1
                p12 = vect1.calculate_point(K1temp)

                for i2, k2n in enumerate(K2):
                    K2temp = np.copy(K2) #Find the probing points
                    K2temp[i2] = k2n + 0.1
                    p21 = vect2.calculate_point(K2temp)
                    K2temp[i2] = k2n -0.1
                    p22 = vect2.calculate_point(K2temp)

                    #Compare probed points
                    all_dist = np.array([self.calc_l2(p11, p21), self.calc_l2(p11,p22), self.calc_l2(p12, p21), self.calc_l2(p12, p22), dist_minimum])
                    result = np.argmin(all_dist)
                    if result == 0:
                        K1update = np.copy(K1)
                        K1update[i1] = k1n + 0.1 #Can maybe update this here to give better guesses.
                        K2update = np.copy(K2)
                        K2update[i2] = k2n + 0.1
                        dist_minimum = all_dist[result]
                    elif result == 1:
                        K1update = np.copy(K1)
                        K1update[i1] = k1n + 0.1
                        K2update = np.copy(K2)
                        K2update[i2] = k2n - 0.1
                        dist_minimum = all_dist[result]
                    elif result == 2:
                        K1update = np.copy(K1)
                        K1update[i1] = k1n - 0.1
                        K2update = np.copy(K2)
                        K2update[i2] = k2n + 0.1
                        dist_minimum = all_dist[result]
                    elif result == 3:
                        K1update = np.copy(K1)
                        K1update[i1] = k1n - 0.1
                        K2update = np.copy(K2)
                        K2update[i2] = k2n - 0.1
                        dist_minimum = all_dist[result]
            
            if ((K1update == K1).all() and (K2update == K2).all()):
                converged = True
            K1 = K1update.copy()
            K2 = K2update.copy()
            iteration_counter = iteration_counter + 1
        
        #print("Final K1 and K2: ")
        #print("K1:")
        #print(K1)
        #print("K2:")
        #print(K2)
        return(K1, K2, iteration_counter)
    

    def calculate_orthogonal(self, pt, vect):
        assert isinstance(pt, np.ndarray), "Point must be in form of a numpy array."
        assert isinstance(vect, Vector), "Vector must be in form of a Vector object."
        assert np.size(pt) == np.size(vect.dirarr[0]), "Vector and pt must exist in same dimensionality."

        #Normalise starting point relative to vect to origin
        pt = np.subtract(pt, vect.dirarr[0])
        
        #Formulate simultaneous equations
        
        #All points in starting point pt multiplied by the directions, summated for each K.
        b = np.zeros(vect.dim)
        for ind in range(np.size(b)):
            b[ind] = ele_multiply_summation(pt, vect.dirarr[ind+1])

        A = np.zeros((vect.dim, vect.dim))

        for row in range(vect.dim):
            for column in range(row, vect.dim):
                result = ele_multiply_summation(vect.dirarr[row+1], vect.dirarr[column+1])
                A[row][column] = result
                A[column][row] = result

        return np.linalg.solve(A, b)#Returns Kvals of closest point on plane to given point.
    

    def orthogonal_descent(self, vect1, vect2):

        K1 = np.zeros(vect1.dim) #Memory allocation
        K2 = np.zeros(vect2.dim)
        K1prev = np.zeros(vect1.dim) #Memory allocation
        K2prev = np.zeros(vect2.dim)

        ctr = 0

        start_pt = vect1.dirarr[0]
        conv = False
        while conv == False:
            K2 = self.calculate_orthogonal(start_pt, vect2)
            start_pt = vect2.calculate_point(K2)
            K1 = self.calculate_orthogonal(start_pt, vect1)
            start_pt = vect1.calculate_point(K1)

            if (rms_diff_norm(K1, K1prev) < 0.01 and rms_diff_norm(K2, K2prev) < 0.01) or ctr > 10000:
                if ctr != 0:
                    conv = True
                if ctr > 10000:
                    print("Exceeded acceptable iterations.")
                    return K1, K2, ctr
            
            K1prev = K1
            K2prev = K2
            ctr = ctr+1

            if ctr % 100 == 0:
                print(f"Current iteration: {ctr}")


        return K1, K2, ctr
    
    def orthogonal_descent_momentum(self, vect1, vect2, beta):

        K1 = np.zeros(vect1.dim) #Memory allocation
        K2 = np.zeros(vect2.dim)
        K1prev = np.zeros(vect1.dim) #Memory allocation
        K2prev = np.zeros(vect2.dim)

        ctr = 1

        start_pt = vect1.dirarr[0]
        conv = False

        #Initialise first iteration
        K2 = self.calculate_orthogonal(start_pt, vect2)
        start_pt = vect2.calculate_point(K2)
        K1 = self.calculate_orthogonal(start_pt, vect1)
        start_pt = vect1.calculate_point(K1)

        K1prev = K1
        K2prev = K2

        mK1 = np.zeros(vect1.dim)
        mK2 = np.zeros(vect2.dim)

        while conv == False:
            K2 = self.calculate_orthogonal(start_pt, vect2)
            dK2 = np.subtract(K2, K2prev)
            mK2 = beta*np.add(mK2, dK2)
            K2 = np.add(K2, mK2)
            start_pt = vect2.calculate_point(K2)
            K1 = self.calculate_orthogonal(start_pt, vect1)
            dK1 = np.subtract(K1, K1prev)
            mK1 = beta*np.add(mK1, dK1)
            K1 = np.add(K1, mK1)
            start_pt = vect1.calculate_point(K1)

            if (rms_diff_norm(K1, K1prev) < 0.01 and rms_diff_norm(K2, K2prev) < 0.01) or ctr > 1000:
                if ctr != 0:
                    conv = True
                if ctr > 1000:
                    print("Exceeded acceptable iterations.")
                    return K1, K2, ctr
                
            
            K1prev = K1
            K2prev = K2
            ctr = ctr+1

            if ctr % 100 == 0:
                print(f"Current iteration: {ctr}")


        return K1, K2, ctr
    
    def orthogonal_descent_basis(self, vect1, vect2):

        vect1 = self.gram_schmidt(vect1)
        vect2 = self.gram_schmidt(vect2)

        K1 = np.zeros(vect1.dim) #Memory allocation
        K2 = np.zeros(vect2.dim)
        K1prev = np.zeros(vect1.dim) #Memory allocation
        K2prev = np.zeros(vect2.dim)

        ctr = 0

        start_pt = vect1.dirarr[0]
        conv = False
        while conv == False:
            K2 = self.sum_linear_projections(start_pt, vect2)
            start_pt = vect2.calculate_point(K2)
            K1 = self.sum_linear_projections(start_pt, vect1)
            start_pt = vect1.calculate_point(K1)

            if (rms_diff_norm(K1, K1prev) < 0.01 and rms_diff_norm(K2, K2prev) < 0.01) or ctr > 10000:
                if ctr != 0:
                    conv = True
                if ctr > 10000:
                    print("Exceeded acceptable iterations.")
                    return K1, K2, ctr
            
            K1prev = K1
            K2prev = K2
            ctr = ctr+1

            if ctr % 100 == 0:
                print(f"Current iteration: {ctr}")
        
        return K1, K2, ctr
    
    def orthogonal_descent_momentum_basis(self, vect1, vect2, beta):

        K1 = np.zeros(vect1.dim) #Memory allocation
        K2 = np.zeros(vect2.dim)
        K1prev = np.zeros(vect1.dim) #Memory allocation
        K2prev = np.zeros(vect2.dim)

        ctr = 1

        start_pt = vect1.dirarr[0]
        conv = False

        vect1 = self.gram_schmidt(vect1)
        vect2 = self.gram_schmidt(vect2)

        #Initialise first iteration
        K2 = self.sum_linear_projections(start_pt, vect2)
        start_pt = vect2.calculate_point(K2)
        K1 = self.sum_linear_projections(start_pt, vect1)
        start_pt = vect1.calculate_point(K1)

        K1prev = K1
        K2prev = K2

        mK1 = np.zeros(vect1.dim)
        mK2 = np.zeros(vect2.dim)

        while conv == False:
            K2 = self.sum_linear_projections(start_pt, vect2)
            dK2 = np.subtract(K2, K2prev)
            mK2 = beta*np.add(mK2, dK2)
            K2 = np.add(K2, mK2)
            start_pt = vect2.calculate_point(K2)
            K1 = self.sum_linear_projections(start_pt, vect1)
            dK1 = np.subtract(K1, K1prev)
            mK1 = beta*np.add(mK1, dK1)
            K1 = np.add(K1, mK1)
            start_pt = vect1.calculate_point(K1)

            if (rms_diff_norm(K1, K1prev) < 0.01 and rms_diff_norm(K2, K2prev) < 0.01) or ctr > 10000:
                if ctr != 0:
                    conv = True
                if ctr > 10000:
                    print("Exceeded acceptable iterations.")
                    return K1, K2, ctr
                
            
            K1prev = K1
            K2prev = K2
            ctr = ctr+1

            if ctr % 100 == 0:
                print(f"Current iteration: {ctr}")

        return K1, K2, ctr
    
    def energy_descent(self, vect1, vect2):

        K1 = np.zeros(vect1.dim) #Memory allocation
        K2 = np.zeros(vect2.dim)
        K1prev = np.zeros(vect1.dim) #Memory allocation
        K2prev = np.zeros(vect2.dim)

        ctr = 1

        start_pt = vect1.dirarr[0]
        conv = False

        vect1 = self.gram_schmidt(vect1)
        vect2 = self.gram_schmidt(vect2)

        #Initialise first iteration
        K2 = self.sum_linear_projections(start_pt, vect2)
        start_pt = vect2.calculate_point(K2)
        K1 = self.sum_linear_projections(start_pt, vect1)
        start_pt = vect1.calculate_point(K1)

        K1prev = K1
        K2prev = K2

        mK1 = np.zeros(vect1.dim)
        mK2 = np.zeros(vect2.dim)

        while conv == False:
            K2 = self.sum_linear_projections(start_pt, vect2)
            dK2 = np.subtract(K2, K2prev)
            print(f"Here! {dK2}")
            K2 = np.add(K2, mK2)
            mK2Eprev = calculate_magnitude(mK2)
            mK2 = np.add(mK2, dK2) #Problem here!!!!! Ordering - momentum is added first, where dK1 is calculated, and then K1 is changed based on this, meaning that K1 is always changed by double what it should be.
            mK2Eaft = calculate_magnitude(mK2)
            if mK2Eaft < mK2Eprev:
                mK2 = np.zeros(vect2.dim)
                print(f"mK2 energy before: {mK2Eprev}")
                print(f"mK2 energy after:  {mK2Eaft}")
                print(f"mK2: {mK2}")
                print(f"dK2: {dK2}")
                print(f"K2: {K2}")
            else:
                print(f"mK2 energy before: {mK2Eprev}")
                print(f"mK2 energy after:  {mK2Eaft}")
                print(f"mK2: {mK2}")
                print(f"dK2: {dK2}")
                print(f"K2: {K2}")

            start_pt = vect1.calculate_point(K2)

            K1 = self.sum_linear_projections(start_pt, vect1)
            dK1 = np.subtract(K1, K1prev)
            K1 = np.add(K1, mK1)
            mK1Eprev = calculate_magnitude(mK1)
            mK1 = np.add(mK1, dK1) #Problem here!!!!! Ordering - momentum is added first, where dK1 is calculated, and then K1 is changed based on this, meaning that K1 is always changed by double what it should be.
            mK1Eaft = calculate_magnitude(mK1)
            if mK1Eaft < mK1Eprev:
                mK1 = np.zeros(vect1.dim)
                print(f"mK1 energy before: {mK1Eprev}")
                print(f"mK1 energy after:  {mK1Eaft}")
                print(f"mK1: {mK1}")
                print(f"dK1: {dK1}")
                print(f"K1: {K1}")
            else:
                print(f"mK1 energy before: {mK1Eprev}")
                print(f"mK1 energy after:  {mK1Eaft}")
                print(f"mK1: {mK1}")
                print(f"dK1: {dK1}")
                print(f"K1: {K1}")
                
            
            
            start_pt = vect1.calculate_point(K1)

            if (rms_diff_norm(K1, K1prev) < 0.01 and rms_diff_norm(K2, K2prev) < 0.01) or ctr > 10:
                if ctr != 0:
                    conv = True
                if ctr > 10:
                    print("Exceeded acceptable iterations.")
                    return K1, K2, ctr
                
            
            K1prev = K1
            K2prev = K2
            ctr = ctr+1

            if ctr % 100 == 0:
                print(f"Current iteration: {ctr}")

        return K1, K2, ctr
    
    def energy_descent_2(self, vect1, vect2, beta_init, scale_factor):

        K1 = np.zeros(vect1.dim) #Memory allocation
        K2 = np.zeros(vect2.dim)
        K1prev = np.zeros(vect1.dim) #Memory allocation
        K2prev = np.zeros(vect2.dim)
        mK1 = np.zeros(vect1.dim) #Memory allocation
        mK2 = np.zeros(vect2.dim)

        ctr = 1
        beta = beta_init

        start_pt = vect1.dirarr[0]
        conv = False

        #Orthogonalise basis
        vect1 = self.gram_schmidt(vect1)
        vect2 = self.gram_schmidt(vect2)

        #Initialise first iteration
        K2 = self.sum_linear_projections(start_pt, vect2)
        start_pt = vect2.calculate_point(K2)
        K1 = self.sum_linear_projections(start_pt, vect1)
        start_pt = vect1.calculate_point(K1)

        K1prev = K1
        K2prev = K2

        ploss = np.sum(np.subtract(vect1.calculate_point(K1), vect2.calculate_point(K2))**2)

        while conv == False:
            
            K2 = self.sum_linear_projections(start_pt, vect2) #Calculate K2 from local point
            dK2 = np.subtract(K2, K2prev) #Calculate organic change in dK2
            K2 = np.add(K2, mK2) #Add momentum
            mK2 = beta*np.add(mK2, dK2) #Update momentum
            start_pt = vect2.calculate_point(K2) #Update starting point

            K1 = self.sum_linear_projections(start_pt, vect1) #Calculate K1 from local point
            dK1= np.subtract(K1, K1prev) #Calculate organic change in dK1
            K1 = np.add(K1, mK1) #Add momentum
            mK1 = beta*np.add(mK1, dK1) #Update momentum
            start_pt = vect1.calculate_point(K1) #Update starting point

            #Check for convergence
            if (rms_diff_norm(K1, K1prev) < 0.01 and rms_diff_norm(K2, K2prev) < 0.01) or ctr > 10000:
                if ctr != 0:
                    conv = True
                if ctr > 10000:
                    print("Exceeded acceptable iterations.")
                    return K1, K2, ctr

            #One "iteration" to guage loss
            tK2 = self.sum_linear_projections(start_pt, vect2)
            start_pt = vect2.calculate_point(tK2)
            tK1 = self.sum_linear_projections(start_pt, vect1)
            start_pt = vect1.calculate_point(tK1)

            nloss = np.sum(np.subtract(vect1.calculate_point(tK1), vect2.calculate_point(tK2))**2)

            if nloss > ploss: #If new result is worse
                mK1 = np.zeros(vect1.dim) #Reset momentum values to zero
                mK2 = np.zeros(vect2.dim)
                K1 = K1prev #Reset K1 and K2
                K2 = K2prev
                beta = beta * scale_factor #Exponentially decreases momentum hyperparameter as overshoot occurs

            else: #If new result is better
                K1prev = K1
                K2prev = K2
                ploss = nloss

            if ctr % 100 == 0:
                print(f"Current iteration: {ctr}")
            
            ctr = ctr + 1

        return K1, K2, ctr


    def quadratic_solve(self, vect1, vect2):
        results = []
        results_iter = np.array([1, 2, 3])
        for iter in range(1, 4):
            K1 = np.array([iter])
            K2 = self.calculate_orthogonal(vect1.calculate_point(K1), vect2)
            l2norm = self.calc_l2(vect2.calculate_point(K2), vect1.calculate_point(K1))
            results.append(l2norm)
        a, b, c = solve_quadratic_2(results_iter, np.array(results))
        K1 = np.array([find_2d_vals(a, b)])
        K2 = np.array([self.calculate_orthogonal(vect1.calculate_point(K1), vect2)])
        return K1, K2
    
    def gram_schmidt(self, vect):
        assert isinstance(vect, Vector), "gram_schmidt must take a Vector class object."
        
        dim = np.shape(vect.dirarr)[0]
        if dim <= 2:
            print("0 or 1D k-plane, already orthogonalised basis.")
            return vect

        for current_col in range (2, dim):

            proj_total = vect.dirarr[current_col]
            for all_prev_cols in range(1, current_col):
                proj_total = calc_orthog_basis(proj_total, vect.dirarr[all_prev_cols])
            vect.dirarr[current_col] = normalise(proj_total)

        return vect
    
    def sum_linear_projections(self, pt, vect): #Must take a NORMALISED orthogonal basis or near-orthogonal, i.e, pushed through gram_schmidt already. Returns kvals
        assert isinstance(vect, Vector), "sum_linear_projections must take a Vector object"
        assert isinstance(pt, np.ndarray), "sum_linear_projections must take a numpy array"

        shift_pt = np.subtract(pt, vect.dirarr[0])
        #proj_tot = np.zeros(self.dim)
        kvals = []
        for col in range(1, np.shape(vect.dirarr)[0]):
            kval = np.dot(shift_pt, vect.dirarr[col])
            #proj_tot = np.add(proj_tot, np.multiply(kval, vect.dirarr[col]))
            kvals.append(kval)
        
        return np.array(kvals)

    def avg_projections(self, vectarr): #Takes an array of Vector objects, iterates to NPA between all vector spaces

        #number of vectors
        vectnum = np.size(vectarr)

        #convergence bool
        converged = False

        for i in range(vectnum):
            vectarr[i] = self.gram_schmidt(vectarr[i])

        #Allocate list memory
        karr = [None] * vectnum

        
        for i in range(vectnum):
            karr[i] = np.zeros(vectarr[i].dim)

        #initial point
        pt = np.array([2, 72])

        ctr = 0
        
        while converged == False:
            tot = np.zeros(vectarr[0].dim)
            current_mag = 0
            for i in range(vectnum):
                proj = vectarr[i].calculate_point(self.sum_linear_projections(pt, vectarr[i]))
                tot = np.add(tot, proj)
                current_mag = current_mag + calculate_magnitude(np.subtract(pt,proj))
            print(current_mag)
            tot = np.divide(tot, vectnum)

            pt = tot

            ctr = ctr + 1

            if ctr == 100:
                converged = True
        
        return pt


            




class Vector:
    def __init__(self, dim, dirarr): #All vectors will be in the form (x, y, z) = (x0, y0, z0) + k1(a1, b1, c1) + k2(a2, b2, c2) ... i.e, inital point + direction.
        assert np.shape(dirarr)[0] == dim + 1, f"Direction array doesn't match assigned dimension. dim = {dim}, dirarr has {np.shape[dirarr][0]} rows, should have {dim + 1} rows.."
        assert isinstance(dirarr, np.ndarray), "dirarr must be a numpy array."
        self.dim = dim
        self.dirarr = dirarr #dirarr will be in form [[x0, y0, z0], [a1, b1, c1], [a2, b2, c2] ...]
    
    def calculate_point(self, constarr): #constarr in form [k1, k2, k3...]
        assert isinstance(constarr, np.ndarray), "constarr must be a numpy array."
        assert np.size(constarr) == self.dim, f"Inputted constant array doesn't match vector dimensionality, expected dimensionality {self.dim}, inputted constarr has {np.size(constarr)} elements."
        point = self.dirarr[0]
        for ind in range(0, self.dim):
            point = np.add(point, constarr[ind]*self.dirarr[ind+1])
        return point
    
    def normalise_dir(self):
        for ind in range(1, np.shape(self.dirarr)[0]):
            self.dirarr[ind] = normalise(self.dirarr[ind])

    def calculate_magnitude(self): #Outputs the magnitude of each dimension of the vector - normalised vectors should all output 1.
        dimensions = np.shape(self.dirarr)[0]-1
        output = np.zeros(dimensions)
        for ind in range(dimensions):
            output[ind] = calculate_magnitude(self.dirarr[ind+1])
        return output
            

    

    #Utility
    def print_vector(self):
        part1, part2 = self.dirarr[0], self.dirarr[1:]
        print(f"Vector start point: \n{part1}.\n Vector directions: \n{part2}.")


