# transformer_from_scratch
This is implementation of transformer for self-study

## training
The training was done in the lambda cloud `gpu_1x_gh200`.
```
ubuntu@192-222-58-52:~/project/transformer_from_scratch$ python train.py
cuda
# of tokens in txt: 35323
Tokens seen: 4096
Epoch: 1, Loss: 5.804877758026123
Epoch: 2, Loss: 3.741780771928675
Epoch: 3, Loss: 2.346310349071727
Epoch: 4, Loss: 1.0587690420010512
Epoch: 5, Loss: 0.4345137910807834
Epoch: 6, Loss: 0.2750820754205479
Epoch: 7, Loss: 0.2264650225201074
Epoch: 8, Loss: 0.20463062559857087
Epoch: 9, Loss: 0.1895058748914915
Epoch: 10, Loss: 0.18003187350490513
Epoch: 11, Loss: 0.17463716404402957
Epoch: 12, Loss: 0.1692803055047989
Epoch: 13, Loss: 0.16503132068935564
...
Epoch: 63, Loss: 0.13082189792219331
Epoch: 64, Loss: 0.13135370513533845
Epoch: 65, Loss: 0.1297982819378376
...
Epoch: 99, Loss: 0.12265084300409346
Epoch: 100, Loss: 0.12297938555917319
```

## result
### alice in wonderland
```
Start context:  We're all
0 : We're all mad here. I'm mad. You're mad.' 'How do you know I'm mad?' said Alice. 'You must be,' said the Cat, 'You must be,' said the Cat, 'or you might bite,' said the Cat
1 : We're all mad here. I'm mad. You're mad.' 'How do you know I'm mad?' said Alice. 'You must be,' said the Cat, 'You must be,' said the Cat, 'or you might bite,' said the Cat
2 : We're all mad here. I'm mad. You're mad.' 'How do you know I'm mad?' said Alice. 'You must be,' said the Cat, 'You must be,' said the Cat, 'or you can't talk about, '
3 : We're all wrong.' 'Yes, but I grow at a reasonable pace,' said the Dormouse: 'not in that ridiculous fashion.' And he got up very grave such a minute or two, But she added in to the time he got up very grave
4 : We're all wrong!' cried the Mock Turtle, capering wildly about. 'Change lobsters again!' yelled the Gryphon at the top of its voice. 'Back to itself, and the Mock Turtle. 'Back to its voice. 'Back to itself
```

### harry potter 02
```
Start context:  gryffindor
0 : gryffindor, I suggest you look more closely at this.â€ Dumbledore reached across to Professor McGonagallâ€™s desk, looking around to say, but thought Harryâ€™s desk, when he kicked out of the bird looked
1 : gryffindor Tower, desperate to tell Ron and Hermione about Colin and Dobby, but they werenâ€™t there. Harry left to look for them so far too many questions for them, wondering if only when heâ€ He pulled him out
2 : gryffindor, I suggest you look more closely at this.â€ Dumbledore reached across to Professor McGonagallâ€™s desk, picked up the desk, picked up the hat, picked up the hat, the hat next to Hedwigâ
3 : gryffindor, occasionally coming to long enough to copy down a name or date, then falling asleep again. He had been speaking for half an hour when something â€™s idea if it in case you know about six oâ€”â€
4 : gryffindor Tower. The castle was quiet; it seemed that the feast was over. They walked past muttering portraits and creaking suits of armor, and climbed narrow flights and climbed narrow windows of armor, examining theICE Peeves broke into the pool of
```
