import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, '..')
from utils.data_utils import *


MAX_DEPTH = 5
INPUT_SIZE = (MAX_DEPTH + 1)*(MAX_DEPTH + 2) + 1 + 1

PREFIX_SIZE = 250



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encode_vectors = nn.Linear(
            MAX_TAGS,
            MAX_TAGS,
            bias=True,
        )
        self.encode_first_part = nn.Linear(
            33,
            15
        )

        self.encode_third_part = nn.Linear(
            110, 
            50
        )

        self.encode_write = nn.Linear(
            43,
            20
        )

        self.encode_access = nn.Linear(
            44,
            20
        )

        self.accesses_embed = nn.Linear(
            300, 
            150
        )

        # LSTM to encode computations
        self.transformation_vectors_embed = nn.LSTM(
            input_size = MAX_TAGS,
            hidden_size = 100,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
        )
        # LSTM to encode computation expressions
        self.exprs_embed = nn.LSTM(
            input_size = 11,
            hidden_size = 100,
            batch_first=True,
        )
        """
        self.fixed_fc = nn.Sequential(
            nn.Linear(535,450),
            nn.ELU(),
            nn.Dropout(0.1)
        )
        """

        self.big_embed = nn.Sequential(
            nn.Linear(535, 450),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(450, 350),
            #nn.ELU(),
            nn.Dropout(0.05),
            #nn.Linear(350, 250),
            #nn.Dropout(0.05),
        )


    def forward(self, x):

        # third part includes loop and write access
        first_part, vectors, third_part = seperate_vector(x[:, :910])

        first_part = self.encode_first_part(first_part)

        vectors = self.encode_vectors(vectors)
        _, (prog_embedding, _) = self.transformation_vectors_embed(vectors)
        prog_embedding = prog_embedding.view((-1, 200))
        
        read_accesses = third_part[:, 153:].view((-1, 15, 44))
        read_accesses = self.encode_access(read_accesses).view(-1, 300)
        read_embedding = self.accesses_embed(read_accesses)
         
        write_access = third_part[:, 110:153]
        write_access = self.encode_write(write_access)

        third_part = self.encode_third_part(third_part[:, :110]) # iteration domain
       
        _, (expr_embedding, _) = self.exprs_embed(x[:, 910:].view((-1, 66, 11)))
        expr_embedding = expr_embedding.view((-1, 100))

        
        # Concatinate the leftover parts from the computatuion, the vectors embedding, and the expression embedding

        x = torch.cat(
            (
                first_part,
                prog_embedding,
                third_part,
                write_access,
                read_embedding,
                expr_embedding,
            ),
            dim=1,
        )

        #x = self.fixed_fc(x)
        return self.big_embed(x)
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            #nn.Linear(250, 350), # <-
            #nn.ELU(),
            #nn.Dropout(0.05),
            nn.Linear(350, 600), # <-
            nn.ELU(),
            nn.Dropout(0.05),
            nn.Linear(600, 1200), # <-
            nn.ELU(),
            nn.Dropout(0.05),
            nn.Linear(1200, 1636) # <-
        )

    def forward(self, x):
        return self.fc(x)

    
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = Encoder()

        #self.regularization = nn.Dropout(0.05)  # Adjust the dropout rate as needed

        # Decoder
        self.decoder = Decoder()

    def forward(self, inputs):
        codes = self.encoder(inputs)
        #dropped = self.regularization(codes)
        decoded = self.decoder(codes)
        return decoded
    
