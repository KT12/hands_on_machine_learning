1) Seq to seq RNN: machine translation, weather prediction
Seq to vector RNN: sentiment analyzer, collaborative filtering
Vec to seq RNN: picture captioning, categorization

2) Encoder-decoder RNN's are used for machine translation because words must be translated within the context of their sentence.
A sequence to sequence RNN would take each word and translate it by itself.

3) A convnet could be combined with an RNN to classify videos.  The convent would examine each still frame to identify features.  The RNN could use LSTM cells to examine how those images change over time.

4) The	 dynamic_rnn() 	function	uses	a	 while_loop() 	operation	to	run	over	the	cell	the	appropriate	number
of	times,	and	you	can	set	 swap_memory=True 	if	you	want	it	to	swap	the	GPU’s	memory	to	the	CPU’s
memory	during	backpropagation	to	avoid	OOM	errors.	Conveniently,	it	also	accepts	a	single	tensor	for	all
inputs	at	every	time	step	(shape	 [None,	n_steps,	n_inputs] )	and	it	outputs	a	single	tensor	for	all
outputs	at	every	time	step	(shape	 [None,	n_steps,	n_neurons] );	there	is	no	need	to	stack,	unstack,	or
transpose

5) Variable length input sequences can be dealt with by padding the input with zeros.
Variable length output sequences can be dealt with by defining a special end of sequence token.

6) You can distribute training and execution of deep RNN by placing each layer on a different GPU.