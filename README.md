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
Start context:  shut up
0 : shut up, Malfoy,” said Ron, bent double with his head in a peony bush, “like fat little Santa Clauses with fishing rods.…’Course,” There was as long enough.…” There was visible
1 : shut up, Malfoy.” “You’re just jealous,” piped up Colin, whose entire body was about as thick as Crabbe’s Cross at him. He could reach. “What’s vast,
2 : shut up, Malfoy.” “You’re just jealous,” piped up Colin, whose entire body was about as thick as Crabbe’s Cross at him. “What” Heir of interest. He could
3 : shut up, Malfoy.” “You’re just jealous,” piped up Colin, whose entire body was about as thick as Crabbe” Heir by sight, yet closer. “What happened to reach Malfoy’
4 : shut up, Malfoy.” “You’re just jealous,” piped up Colin, whose entire body was about as thick as Crabbe’s Cross at him. “Now, and Nearly Headless Hunt, by common
5 : shut up, Malfoy.” “You’re just jealous,” piped up Colin, whose entire body was about as thick as Crabbe’s Cross at him. “Jig of what Ginny’s bank,
6 : shut up, Malfoy,” said Ron, bent double with his head in a peony bush, “like fat little Santa Clauses with fishing rods.…’s family but oh,” said,’you.…’
7 : shut up, Malfoy.” “You’re just jealous,” piped up Colin, whose entire body was about as thick as Crabbe’s neck. “What’s rage did try and scared as Crabbe
8 : shut up, Malfoy,” said Ron, bent double with his head in a peony bush, “like fat little Santa Clauses with fishing rods.…’Course,” There was as long,’ then he heard more difficult
9 : shut up, Malfoy.” “You’re just jealous,” piped up Colin, whose entire body was about as thick as Crabbe’s trunk. He focused on the mud, Ron’s neck. He could
```
