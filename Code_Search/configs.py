
def config_JointEmbeder():   
    conf = {
        # data_params
            'dataset_name': 'CodeSearchDataset',  # name of dataset to specify a data loader
            #training data
            'train_name': 'train.methname.h5',
            'train_code': 'train.code.h5',
            'train_desc': 'train.desc.h5',
            #test data
            'valid_name': 'test.methname.h5',
            'valid_desc': 'test.desc.h5',
            #use data (computing code vectors)
            'use_codebase': 'use.rawcode.txt',#'use.rawcode.h5'
            'use_names': 'use.methname.CSN.h5',
            'use_codes': 'use.code.CSN.h5',
            #results data(code vectors)
            'use_codevecs': 'use.codevecs.normalized.h5',#'use.codevecs.h5',
                   
            #parameters
            'name_len': 6,
            'api_len': 30,
            'code_len': 50,
            'desc_len': 30,
            'n_words': 10000,  # len(vocabulary) + 1
            #vocabulary info
            'vocab_name': 'vocab.name.json',
            'vocab_code': 'vocab.code.json',
            'vocab_desc': 'vocab.desc.json',
                    
        #training_params            
            'batch_size': 64,
            'chunk_size': 100000,
            'nb_epoch': 500,
            #'optimizer': 'adam',
            'learning_rate': 1e-4, #1e-4
            'adam_epsilon': 1e-8,
            'warmup_steps': 5000,
            'fp16': False,
            'fp16_opt_level': 'O1', #For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].
                            #"See details at https://nvidia.github.io/apex/amp.html"

        # model_params
            'emb_size': 512, #128
            'n_hidden': 512,#128 number of hidden dimension of code/desc representation
            # recurrent
            'lstm_dims': 256, #128 * 2
            'init_embed_weights_name': None,#'word2vec_100_methname.h5', 
            'init_embed_weights_tokens': None,#'word2vec_100_tokens.h5', 
            'init_embed_weights_desc': None,#'word2vec_100_desc.h5',           
            'margin': 0.3986,
            'sim_measure': 'cos',#similarity measure: gesd, cosine, aesd
         
    }
    return conf


