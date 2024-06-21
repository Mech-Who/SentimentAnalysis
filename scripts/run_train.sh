# # rnn simple cell
# python lstm/other_train.py -m rnn_simple_cell -s ./model > logs/train_rnn_simple.log 2>&1 &
# wait
# # rnn gru cell
# python lstm/other_train.py -m rnn_gru_cell -s ./model -d ./data > logs/train_rnn_gru.log 2>&1 &
# wait
# # rnn lstm cell
# python lstm/other_train.py -m rnn_lstm_cell -s ./model -d ./data > logs/train_rnn_lstm.log 2>&1 &
# wait
# # simple rnn
# python lstm/other_train.py -m simple_rnn -s ./model -d ./data > logs/train_simple_rnn.log 2>&1 &
# wait
# # gru
# python lstm/other_train.py -m gru -s ./model -d ./data > logs/train_gru.log 2>&1 &
# wait
# # conv lstm
# python lstm/other_train.py -m convlstm -s ./model -d ./data > logs/train_convlstm.log 2>&1 &
# wait
# # lstm
# python lstm/other_train.py -m lstm -s ./model -d ./data > logs/train_lstm.log 2>&1 &
# wait
# bi-lstm + attention
python lstm/other_train.py -m bidirectional_lstm -a attention -s ./model -d ./data > logs/train_bi_lstm_with_attention.log 2>&1 &
wait
# bi-lstm + self-attention
python lstm/other_train.py -m bidirectional_lstm -a self-attention -s ./model -d ./data > logs/train_bi_lstm_with_self_attention.log 2>&1 &
wait
# bi-gru + attention
python lstm/other_train.py -m bidirectional_gru -a attention -s ./model -d ./data > logs/train_bi_gru_with_attention.log 2>&1 &
wait
# bi-gru + self-attention
python lstm/other_train.py -m bidirectional_gru -a self-attention -s ./model -d ./data > logs/train_bi_gru_with_self_attention.log 2>&1 &
