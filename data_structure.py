class Data():
    def __init__(self, Y, H, std, curr_time, prev_time):
        self.Y = Y
        self.H = H.T
        self.std = std
        self.curr = curr_time
        self.prev = prev_time
        self.batch_ind = 0;
    
    def getY(self):
        return self.Y
    
    def getH(self):
        return self.H
    
    def getStdDev(self):
        return self.std
    
    def completed(self):
        return self.batch_ind == self.Y.size
    
    #returns R and not s
    def getNextBatch(self):
        self.batch_ind += 1
        return self.H[self.batch_ind - 1].reshape(1,4), self.Y[self.batch_ind - 1][0], self.std[self.batch_ind - 1][0]**2
    
    def getDeltaTime(self):
        return self.curr - self.prev
    
    def getTimes(self):
        return self.curr, self.prev
    
    def getAverageTime(self):
        return (self.curr+self.prev)/2