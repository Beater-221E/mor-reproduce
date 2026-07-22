# SFT task examples (decoded)


## SidSFTDataset

### example 0

**prompt:**
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

### User Input: 
The user has interacted with items <a_79><b_163><c_3>, <a_97><b_2><c_1>, <a_9><b_163><c_0>, <a_42><b_31><c_7>, <a_139><b_99><c_5>, <a_45><b_181><c_4>, <a_42><b_118><c_0>, <a_251><b_212><c_2>, <a_198><b_196><c_5>, <a_198><b_85><c_2> in chronological order. Can you predict the next possible item that the user may expect?

### Response:

```

**response:**
```
<a_169><b_82><c_210>
<|endoftext|>
```

### example 1

**prompt:**
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

### User Input: 
The user has interacted with items <a_111><b_248><c_24> in chronological order. Can you predict the next possible item that the user may expect?

### Response:

```

**response:**
```
<a_1><b_49><c_3>
<|endoftext|>
```

### example 2

**prompt:**
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

### User Input: 
The user has interacted with items <a_72><b_150><c_0> in chronological order. Can you predict the next possible item that the user may expect?

### Response:

```

**response:**
```
<a_57><b_74><c_6>
<|endoftext|>
```


## SidItemFeatDataset

### example 0

**prompt:**
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Answer the question about item identification.

### User Input: 
Which item has the title: Articulating Borescope Inspection Camera with 5" IPS Screen Industrial Steering Endoscope, 8 Adjustable LEDs, 0.33inch Len HD 1080P Flexible Cable Snake Endoscope Inspection Camera, 3000mAh, 5FT?

### Response:

```

**response:**
```
<a_16><b_76><c_21>
<|endoftext|>
```

### example 1

**prompt:**
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Answer the question about item identification.

### User Input: 
What is the title of item "<a_11><b_212><c_0>"?

### Response:

```

**response:**
```
mxuteuk AC 220V/380V Push Button Switch On/Off Start Stop Switch 3 Phase Self Lock Electric Motor Switches Mechanical Equipment Control Station 10A BS230B
<|endoftext|>
```

### example 2

**prompt:**
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Answer the question about item identification.

### User Input: 
What is the title of item "<a_169><b_46><c_5>"?

### Response:

```

**response:**
```
CO-RODE Tact Button Switch 6x6x5mm Pack of 100
<|endoftext|>
```


## FusionSeqRecDataset

### example 0

**prompt:**
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you recommend the next item for the user based on their interaction history?

### User Input: 
The user has sequentially interacted with items <a_79><b_163><c_3>, <a_97><b_2><c_1>, <a_9><b_163><c_0>, <a_42><b_31><c_7>, <a_139><b_99><c_5>, <a_45><b_181><c_4>, <a_42><b_118><c_0>, <a_251><b_212><c_2>, <a_198><b_196><c_5>, <a_198><b_85><c_2>. Can you recommend the next item for him? Tell me the title of the item

### Response:

```

**response:**
```
Gikfun 6x6x4.3mm TACT Switch Push Button for Arduino PCB (Pack of 50pcs) EK1019
<|endoftext|>
```

### example 1

**prompt:**
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you recommend the next item for the user based on their interaction history?

### User Input: 
The user has sequentially interacted with items <a_111><b_248><c_24>. Can you recommend the next item for him? Tell me the title of the item

### Response:

```

**response:**
```
Amazon Basics TPU 3D Printer Filament, 1.75mm, Red, 1 kg Spool
<|endoftext|>
```

### example 2

**prompt:**
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you recommend the next item for the user based on their interaction history?

### User Input: 
The user has sequentially interacted with items <a_72><b_150><c_0>. Can you recommend the next item for him? Tell me the title of the item

### Response:

```

**response:**
```
50 Pcs Zinc Self-Drilling Drywall Anchors with #8 Stainless Steel Screws Kit, 25 Heavy Duty Metal Wall Anchor and 25 Screws
<|endoftext|>
```
