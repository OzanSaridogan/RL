
# Gymnasium ortamında bir directed weighted graph oluştur.
# f fonksiyonu bir edge in weightini real sayılarda tanımlayan bir fonksiyon
# network X kütüphanesini kullan
# NP-HAND NEDİR ? P=NP Ne ifade eder araştır.
# Traveling salesman program (TSP) nedir araştır

import gymnasium as gym
import networkx as nx
import numpy as np

class WeightedDirectedGraph(gym.Env):

    def _init_(self,graph,start_node,goal_node,n_nodes):
        # Define the structure of the environment
        # Define the possible actions of the agent
        # Define the possible states that the agent can be in

        self.start_node = start_node
        self.goal_node = goal_node
        self.current_state = start_node
        self.g = nx.MultiDiGraph()

        self.g.add_nodes_from(range(n_nodes))
        self.g.add_weighted_edges_from(0,1,10)
        self.g.add_weighted_edges_from(1,2,4)
        self.g.add_weighted_edges_from(2,1,7)
        self.g.add_weighted_edges_from(2,3,16)
        self.g.add_weighted_edges_from(2,5,19)
        self.g.add_weighted_edges_from(2,8,3)
        self.g.add_weighted_edges_from(3,4,1)
        self.g.add_weighted_edges_from(4,0,23)
        self.g.add_weighted_edges_from(5,6,12)
        self.g.add_weighted_edges_from(5,7,2)
        self.g.add_weighted_edges_from(6,9,28)
        self.g.add_weighted_edges_from(7,8,31)
        self.g.add_weighted_edges_from(8,9,4)
        

        
        self.action_space = gym.spaces.Discrete(n_nodes)
        self.observation_space = gym.spaces.Discrete(n_nodes)

    def step(self, action):
        #RL agentının çevresiyle olan tek bir adımdaki etkileşimini simüle eder.
        #Ajanın bu adımı sonrasında bulunduğu state i döndürür
        #Adımı sonucunda aldığı reward değerini döndürür
        #Eğer ajan adımı sonucunda goal state e ulaşmışsa done = true değerini döndürür

        action = self.choose_action(self)

        state = self.current_state
        
        state = action

        if self.current_state == self.goal_node:
            reward = 1
            terminated = True
        else:
            reward = 0
            terminated = False
            

            

        return state, reward, terminated 

    def choose_action(self):

        action = np.random.randint(0,self.n_nodes)
        
    def reset(self, *, seed = None, options = None):
        #purpose of this method is to initiate a new episode for an environment
        #Başlangıç konumuna geri döndürür
        




        return super().reset(seed=seed, options=options)   
    
    def _get_obs(self):

        return self.current_state,self.goal_node