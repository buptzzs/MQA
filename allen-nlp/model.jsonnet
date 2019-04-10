{
   'dataset_reader': {
       'type': 'my_qangaroo',
       'token_indexers':{
         'tokens':{
           'type':'single_id',
           'lowercase_tokens': true
         },

       }
   },
   
   'train_data_path': '../data/qangaroo_v1.1/wikihop/train.json',
   'validation_data_path': '../data/qangaroo_v1.1/wikihop/dev.json',
   "iterator": {
    "type": "bucket",
    "sorting_keys": [['supports','list_num_tokens'],["supports", "num_fields"]],
    "batch_size": 2,
    'cache_instances': false,
    'biggest_batch_first': true,
   },
   
   'model': {
       'type':'msprm',
       'text_field_embedder':{
           'token_embedders':{
               'tokens':{
                   'type': 'embedding',
                   'pretrained_file': '../glove.840B.300d.lower.converted.zip',
                   'embedding_dim': 300,
                   'trainable': false
               },
           },
       },
       
       'phrase_layer':{
           'type': 'gru',
           'input_size': 300,
           'hidden_size': 50,
           'num_layers': 1,
           'bidirectional': true
       },
       
       'pq_attention': {
           'type': 'linear',
           'combination': 'x,y,x*y',
           'tensor_1_dim': 100,
           'tensor_2_dim': 100           
       },
       
       'p_selfattention': {
           'type': 'linear',
           'combination': 'x,y,x*y',
           'tensor_1_dim': 200,
           'tensor_2_dim': 200           
        },
        
       'supports_pooling': {
           'type': 'self_attentive',
           'dim': 200
       },
       'query_pooling': {
           'type': 'self_attentive',
           'dim': 100
       },
       'candidates_pooling': {
           'type': 'self_attentive',
           'dim': 100
       },     
       
       'decoder': {
           'type': 'san_decoder',
           'support_dim': 200,
           'query_dim': 100,
           'candidates_dim': 100,
           'num_step': 5,
           'reason_type': 0,
           'reason_dropout_p': 0.4,
           'dropout_p': 0.2,
       }
   },
   
   'trainer': {
       'num_epochs': 50,
       'grad_norm': 5.0,
       'cuda_device': 0,
       
       'learning_rate_scheduler': {
           'type': 'cosine',
           't_initial': 10,
           't_mul': 1,
           'eta_min': 1e-7,
       },
       'optimizer': {
          'type': 'adam',
          'betas': [0.9, 0.999],
          'lr': 5e-4
        },       
       
       'histogram_interval': 10000,
       'summary_interval': 1000,
       'should_log_learning_rate': true,
       
       'moving_average':{
         'type': 'exponential'
       }
              
   }
   
   

}