This repository contains supporting materials for paper `Designing Network Algorithms via Large Language Models`.

### Table of contents

- [Prompts](#prompts)
    - The prompt for state design
    - The prompt for network architecture design 
- [Top Performing States](#top-performing-states)
    - The best state from GPT-3.5 for the FCC dataset
    - The best state from GPT-4 for the FCC dataset
    - The best state from GPT-3.5 for the Starlink dataset
    - The best state from GPT-4 for the Starlink dataset
    - The best state from GPT-3.5 for the 4G dataset
    - The best state from GPT-4 for the 4G dataset
    - The best state from GPT-3.5 for the 5G dataset
    - The best state from GPT-4 for the 5G dataset
- [Top Perfoming Network Architectures](#top-performing-network-architectures)
    - The best network architecture from GPT-3.5 for the FCC dataset
    - The best network architecture from GPT-3.5 for the Starlink dataset
    - The best network architecture from GPT-3.5 for the 4G dataset
    - The best network architecture from GPT-3.5 for the 5G dataset



### Prompts

The prompt we used to design states:

~~~
You are trying to design the state code of a reinforcement learning algorithm for adaptive bit rate.

Adaptive Bit Rate (ABR) is a streaming technology used in multimedia applications, particularly in video streaming, to optimize the delivery of content over networks with varying bandwidth conditions. The main goal of ABR is to provide a smooth and uninterrupted viewing experience for users by dynamically adjusting the quality of the video based on the available network conditions.

The input variables include:

bit_rate_kbps_list:
Historical bit rates in kbps, candidate values can be [300., 750., 1200., 1850., 2850., 4300.]
bit_rate_kbps_list[-1] is the most recent bit rate.
For example, [300, 750, ..., 1200, 1850, 4300, 4300] means the last two chunks we downloaded is 4300kbps.

buffer_size_second_list:
Historical buffer size (buffered video length) in second
buffer_size_second_list[-1] is the most recent buffer size.

delay_second_list:
Historical delay (download time of the chunk) in second
delay_second_list[-1] is the download time of the most recent downloaded chunk.

video_chunk_size_bytes_list:
Historical downloaded video chunk sizes in bytes
video_chunk_size_bytes_list[-1] is the size of the most recent downloaded chunk.
Thus video_chunk_size_bytes_list[-1] / delay_second_list[-1] is the download throughtput of the most recent chunks

next_chunk_bytes_sizes:
The sizes of the next one chunk in different bit rate levels.
For example, this can be [181801, 450283, 668286, 1034108, 1728879, 2354772],
which means the next chunk will be 181801 bytes if we select 300kbps (all_bit_rate_kbps[0]); is 450283 bytes if we select 750kbps(all_bit_rate_kbps[1])
We always have len(next_chunk_bytes_sizes) = len(all_bit_rate_kbps)

video_chunk_remain_num:
How many remaining video chunks there are. It is a single number.

total_chunk_num:
A single number. total_chunk_num is 48 in most cases.

all_bit_rate_kbps:
all_bit_rate_kbps=[300., 750., 1200., 1850., 2850., 4300.] in most cases


```python
import numpy as np

def state_func(
    # historical bit rates in kbps, candidate values can be [300., 750., 1200., 1850., 2850., 4300.]
    # bit_rate_kbps_list[-1] is the most recent bit rate.
    # For example, [300, 750, ..., 1200, 1850, 4300, 4300] means the last two chunks we downloaded is 4300kbps.
    bit_rate_kbps_list,
    # historical buffer size (buffered video length) in second
    # buffer_size_second_list[-1] is the most recent buffer size.
    buffer_size_second_list,
    # historical delay (download time of the chunk) in second
    # delay_second_list[-1] is the download time of the most recent downloaded chunk.
    delay_second_list,
    # historical downloaded video chunk sizes in bytes
    # video_chunk_size_bytes_list[-1] is the size of the most recent downloaded chunk.
    # Thus video_chunk_size_bytes_list[-1] / delay_second_list[-1] is the download throughtput of the most recent chunks
    video_chunk_size_bytes_list,
    # The sizes of the next one chunk in different bit rate levels.
    # For example, this can be [181801, 450283, 668286, 1034108, 1728879, 2354772],
    # which means the next chunk will be 181801 bytes if we select 300kbps (all_bit_rate_kbps[0]); is 450283 bytes if we select 750kbps(all_bit_rate_kbps[1])
    # We always have len(next_chunk_bytes_sizes) = len(all_bit_rate_kbps)
    next_chunk_bytes_sizes,
    # How many remaining video chunks there are. It is a single number
    video_chunk_remain_num,
    # A single number. total_chunk_num is 48 in most cases.
    total_chunk_num,
    # all_bit_rate_kbps=[300., 750., 1200., 1850., 2850., 4300.] in most cases
    all_bit_rate_kbps,
):
    # normal state 1: The normed last bit rate
    normed_last_bit_rate = bit_rate_kbps_list[-1] / float(np.max(all_bit_rate_kbps))
    # normal state 2: The normed last buffer size second (buffered video second)
    buffer_norm_factor = 10.
    normed_last_buffer_size = buffer_size_second_list[-1] / buffer_norm_factor # in 10-second
    # normal state 3: The percentage of the remaining video chunks.
    remaining_chunk_percentage = float(video_chunk_remain_num / total_chunk_num)
    # Finally, the normal states. Each entry in normal_states should be a list.
    normal_states = [
        [normed_last_bit_rate],
        [normed_last_buffer_size],
        [remaining_chunk_percentage],
    ]

    # time series states
    # use 8 as the time series length for time series state 1 and 2
    history_window = 8
    # time series state 1: Estimated throughput in near history
    # use the unit mega byte per second (it is equiv to kilo byte / ms)
    throughput_MBps_list = []
    for i in range(history_window):
        history_chunk_size_bytes = video_chunk_size_bytes_list[-(history_window - i)]
        history_delay_second = delay_second_list[-(history_window - i)]
        throughput_MBps_list.append(history_chunk_size_bytes / 1000. / 1000. / history_delay_second)
    # time series state 2: The normed download time (delay) in near history
    delay_norm_factor = 10.
    normed_delay_list = [x / delay_norm_factor for x in delay_second_list]
    # time series state 3: Treat next chunk sizes in MB as ts states, too. We use Mega Byte since Byte is too large for NN.
    next_chunk_bytes_MB = [x / 1000. / 1000. for x in next_chunk_bytes_sizes]
    # Finally, the time series states. Each entry in timeseries_states should be a list.
    time_series_states = [
        throughput_MBps_list,
        normed_delay_list,
        next_chunk_bytes_MB,
    ]

    # Return the states
    return {
        "normal_states": normal_states,
        "time_series_states": time_series_states,
    }
```

Try to improve the state design for me. Remember:

1. Please keep the function name `state_func`
2. Please keep the function's input variables. Do not add any new inputs or remove any inputs. But you can decide how to use the existing inputs in the parameter list.
3. Please keep the function's output variables. The outputs should always be {"normal_states": normal_states, "time_series_states": time_series_states}, while normal_states and time_series_states are list of list. Specifically every element in normal_states would be a list of values that will be sent to a fully connected network; and every element in time_series_states will be treated as time series and sent to a 1D conv network.
4. Please keep `import numpy`. You can also import scipy, pandas if needed.
5. Please normalize the input properly when you try to add new states. It is always better to have the output states within the range [-1, 1]. Do not directly put kbps, bytes, or kbps / second in the output.

Using the following format to output:

Analysis and ideas:
<try to analyze the current code, the problem, propose ideas, and choose the best ideas>

Code:
```python
<Your improved state design code here>
```
~~~


The prompt we used to design network architecture:
~~~
You are trying to design a network function of a reinforcement learning algorithm for adaptive bit rate.

Adaptive Bit Rate (ABR) is a streaming technology used in multimedia applications, particularly in video streaming, to optimize the delivery of content over networks with varying bandwidth conditions. The main goal of ABR is to provide a smooth and uninterrupted viewing experience for users by dynamically adjusting the quality of the video based on the available network conditions.

We have 3 inputs: normal_input_list, ts_input_list, and action_dim.

Their meanings are:
  - normal_input_list[0]: normed last bit rate. It is a tf.placeholder and has the shape of `[None, 1]`.
  - normal_input_list[1]: normed last buffer size. It is a tf.placeholder and has the shape of `[None, 1]`.
  - normal_input_list[2]: normed remaining chunk percentage. 0% means we have finished playing. 50% means there are 50% of remaining chunks. It is a tf.placeholder and has the shape of `[None, 1]`.
  - ts_input_list[0]: normed throughput MBps. It is the normed throughput numbers in history. It is a tf.placeholder and has the shape of `[None, history_window_n]`.
  - ts_input_list[1]: normed download time list. It is download time of each chunks. It is a tf.placeholder and has the shape of `[None, history_window_n]`.
  - ts_input_list[2]: next chunk sizes in MB. It is the chunk sizes of the next chunk in different bitrates. It is a tf.placeholder and has the shape of `[None, bit_rate_n]`.


A network function example is:
```python
import tensorflow.compat.v1 as tf
import tflearn

def network_func(normal_input_list, ts_input_list, action_dim):
    with tf.variable_scope('actor'):
        normal_features = [
            tflearn.fully_connected(normal_input, 128, activation='relu')
            for normal_input in normal_input_list
        ]
        ts_features = [
            tflearn.flatten(tflearn.conv_1d(
                tf.expand_dims(ts_input, axis=1), 
                128, 1, activation='relu'
            ))
            for ts_input in ts_input_list
        ]
        merged_features = tflearn.merge(normal_features + ts_features, "concat")
        pi_features = tflearn.fully_connected(merged_features, 128, activation='relu')
        pi = tflearn.fully_connected(pi_features, action_dim, activation='softmax')

    with tf.variable_scope('critic'):
        normal_features = [
            tflearn.fully_connected(normal_input, 128, activation='relu')
            for normal_input in normal_input_list
        ]
        ts_features = [
            tflearn.flatten(tflearn.conv_1d(
                tf.expand_dims(ts_input, axis=1), 
                128, 1, activation='relu'
            ))
            for ts_input in ts_input_list
        ]
        merged_features = tflearn.merge(normal_features + ts_features, "concat")
        value_features = tflearn.fully_connected(merged_features, 128, activation='relu')
        value = tflearn.fully_connected(value_features, 1, activation='linear')

    return pi, value
```

The example uses fully connected network to process all states in normal_input_list, and uses time series network to process the states in ts_input_list.

Please notice that:
- You MUST include "import tensorflow.compat.v1 as tf" and "import tflearn" in the beginning of your code. Use tflearn or tensorflow v1 to program the network.
- You MUST use the same function name: `network_func` in your output.
- You MUST use the two outputs `pi, value` in your returning. We strictly follow the actor-critic structure but you can design the intermediate network structure.
- The two outputs: `pi` should be exactly the same dimention with `action_dim`, which can be viewed as a classification. `value` should be exactly a single value which is used to evaluate the states.
- DO NOT change the input: normal_input_list, ts_input_list, action_dim.
- You can change the hidden_num, the intermediate network structure, or use other tensorflow v1 functions to program the network.

Using the following format to output:

Analysis and ideas:
<try to analyze the current code, the problem, propose ideas, and choose the best ideas>

Code:
```python
<Your improved network design code here>
```
~~~

### Top Performing States

The best state from GPT-3.5 for the FCC dataset:

```python
import numpy as np

def state_func(
    bit_rate_kbps_list,
    buffer_size_second_list,
    delay_second_list,
    video_chunk_size_bytes_list,
    next_chunk_bytes_sizes,
    video_chunk_remain_num,
    total_chunk_num,
    all_bit_rate_kbps,
):
    # Normal state 1: The normed last bit rate
    normed_last_bit_rate = bit_rate_kbps_list[-1] / float(np.max(all_bit_rate_kbps))
    # Normal state 2: The normed last buffer size second (buffered video second)
    buffer_norm_factor = 10.
    normed_last_buffer_size = buffer_size_second_list[-1] / buffer_norm_factor
    # Normal state 3: The percentage of the remaining video chunks
    remaining_chunk_percentage = 2.0 * float(video_chunk_remain_num / total_chunk_num) - 1.0
    # Normal state 4: Average throughput over the history window
    history_window = 8
    avg_throughput_MBps = np.mean([video_chunk_size_bytes_list[-i] / 1000. / 1000. / delay_second_list[-i] for i in range(1, history_window+1)])
    # Normal state 5: Variation in throughput over the history window
    throughput_variation = np.std([video_chunk_size_bytes_list[-i] / 1000. / 1000. / delay_second_list[-i] for i in range(1, history_window+1)])
    # Normal state 6: Buffer occupancy rate
    buffer_occ_rate = 2.0 * (1.0 - buffer_size_second_list[-1] / delay_second_list[-1]) - 1.0
    # Finally, the normal states
    normal_states = [
        [normed_last_bit_rate],
        [normed_last_buffer_size],
        [remaining_chunk_percentage],
        [avg_throughput_MBps],
        [throughput_variation],
        [buffer_occ_rate]
    ]
    # Time series state 1: Estimated throughput in the near history
    throughput_MBps_list = [(video_chunk_size_bytes_list[-i] / 1000. / 1000. / delay_second_list[-i]) for i in range(1, history_window+1)]
    # Time series state 2: The normed download time (delay) in the near history
    delay_norm_factor = 10.
    normed_delay_list = [(delay_second_list[-i] / delay_norm_factor) for i in range(1, history_window+1)]
    # Time series state 3: Treat next chunk sizes in MB as time series states
    next_chunk_bytes_MB = [x / 1000. / 1000. for x in next_chunk_bytes_sizes]
    # Finally, the time series states
    time_series_states = [
        throughput_MBps_list,
        normed_delay_list,
        next_chunk_bytes_MB
    ]
    # Return the states
    return {
        "normal_states": normal_states,
        "time_series_states": time_series_states,
    }
```

The best state from GPT-4 for the FCC dataset:

```python
import numpy as np

def state_func(
    bit_rate_kbps_list,
    buffer_size_second_list,
    delay_second_list,
    video_chunk_size_bytes_list,
    next_chunk_bytes_sizes,
    video_chunk_remain_num,
    total_chunk_num,
    all_bit_rate_kbps,
):
    # Constants
    buffer_norm_factor = 10.
    delay_norm_factor = 10.
    history_window = 8
    # Min-Max normalization for the last bit rate and buffer size
    normed_last_bit_rate = (bit_rate_kbps_list[-1] - min(all_bit_rate_kbps)) / (max(all_bit_rate_kbps) - min(all_bit_rate_kbps)) * 2 - 1
    normed_last_buffer_size = (buffer_size_second_list[-1] / buffer_norm_factor - 0.5) * 2
    # Normal state: Percentage of the remaining video chunks
    remaining_chunk_percentage = video_chunk_remain_num / total_chunk_num * 2 - 1
    # Calculate change rate for the buffer size
    if len(buffer_size_second_list) > 1:
        buffer_change_rate = buffer_size_second_list[-1] - buffer_size_second_list[-2]
    else:
        buffer_change_rate = 0
    normed_buffer_change_rate = buffer_change_rate / buffer_norm_factor * 2  # Assuming change rate won't exceed buffer_norm_factor
    # Normal states
    normal_states = [
        [normed_last_bit_rate],
        [normed_last_buffer_size],
        [remaining_chunk_percentage],
        [normed_buffer_change_rate],
    ]
    # Time series states
    throughput_MBps_list = []
    normed_delay_list = []
    for i in range(history_window):
        # Throughput calculations
        history_chunk_size_bytes = video_chunk_size_bytes_list[-(history_window - i)]
        history_delay_second = delay_second_list[-(history_window - i)]
        throughput = history_chunk_size_bytes / 1000. / 1000. / history_delay_second
        throughput_MBps_list.append(throughput)
        # Normalized delays
        delay = delay_second_list[-(history_window - i)]
        normed_delay = (delay / delay_norm_factor - 0.5) * 2
        normed_delay_list.append(normed_delay)
    # Apply exponential moving average to time series states
    alpha = 0.5  # Smoothing factor
    smoothed_throughput_MBps_list = np.convolve(throughput_MBps_list, [1-alpha, alpha], 'valid').tolist()
    smoothed_normed_delay_list = np.convolve(normed_delay_list, [1-alpha, alpha], 'valid').tolist()
    # Next chunk sizes normalization
    max_chunk_size_MB = max(next_chunk_bytes_sizes) / 1000. / 1000.
    next_chunk_bytes_MB = [(x / 1000. / 1000. / max_chunk_size_MB * 2 - 1) for x in next_chunk_bytes_sizes]
    # Time series states
    time_series_states = [
        smoothed_throughput_MBps_list,
        smoothed_normed_delay_list,
        next_chunk_bytes_MB,
    ]
    # Return the states
    return {
        "normal_states": normal_states,
        "time_series_states": time_series_states,
    }
```

The best state from GPT-3.5 for the Starlink dataset:

```python
import numpy as np

def state_func(
    bit_rate_kbps_list,
    buffer_size_second_list,
    delay_second_list,
    video_chunk_size_bytes_list,
    next_chunk_bytes_sizes,
    video_chunk_remain_num,
    total_chunk_num,
    all_bit_rate_kbps,
):
    # Current state features
    normed_last_bit_rate = bit_rate_kbps_list[-1] / float(np.max(all_bit_rate_kbps))
    buffer_norm_factor = 10.
    normed_last_buffer_size = buffer_size_second_list[-1] / buffer_norm_factor
    remaining_chunk_percentage = float(video_chunk_remain_num / total_chunk_num)
    # New state features
    # Normalized bitrate change over the last few chunks
    bit_rate_change = (bit_rate_kbps_list[-1] - bit_rate_kbps_list[-2]) / float(np.max(all_bit_rate_kbps))
    # Network throughput in near history
    network_throughput_MBps = []
    for i in range(len(bit_rate_kbps_list)):
        throughput = video_chunk_size_bytes_list[i] / 1000. / 1000. / delay_second_list[i]
        network_throughput_MBps.append(throughput)
    # Normalized new state features
    normed_bit_rate_change = bit_rate_change / float(np.max(all_bit_rate_kbps))
    normed_network_throughput = [x / (1000.0 * np.max(all_bit_rate_kbps)) for x in network_throughput_MBps]
    normal_states = [
        [normed_last_bit_rate, normed_last_buffer_size, remaining_chunk_percentage, normed_bit_rate_change]
    ]
    time_series_states = [
        normed_network_throughput
    ]
    return {
        "normal_states": normal_states,
        "time_series_states": time_series_states,
    }
```

The best state from GPT-4 for the Starlink dataset:

```python
import numpy as np

def state_func(
    bit_rate_kbps_list,
    buffer_size_second_list,
    delay_second_list,
    video_chunk_size_bytes_list,
    next_chunk_bytes_sizes,
    video_chunk_remain_num,
    total_chunk_num,
    all_bit_rate_kbps,
):
    # Normalization and state calculation constants
    buffer_norm_factor = 60.  # Increase to 60 seconds or consider a logarithmic approach
    throughput_norm_factor = np.max(all_bit_rate_kbps) / 1000.  # Normalize by max bitrate in Mbps
    delay_norm_factor = 5.  # Increase the normalization factor for delay
    alpha = 0.5  # Exponential averaging factor
    history_window = 8
    # normal states
    normed_last_bit_rate = bit_rate_kbps_list[-1] / float(np.max(all_bit_rate_kbps))
    normed_last_buffer_size = np.log(buffer_size_second_list[-1] + 1) / np.log(buffer_norm_factor)  # Logarithmic scaling
    remaining_chunk_percentage = float(video_chunk_remain_num / total_chunk_num)
    normal_states = [
        [normed_last_bit_rate],
        [normed_last_buffer_size],
        [remaining_chunk_percentage],
    ]
    # time series states
    throughput_MBps_list = [video_chunk_size_bytes_list[-1] / 1000. / 1000. / delay_second_list[-1] / throughput_norm_factor]
    normed_delay_list = [delay_second_list[-1] / delay_norm_factor]
    for i in range(1, history_window):
        # Exponentially averaged throughput and delay
        exp_avg_throughput = alpha * (video_chunk_size_bytes_list[-(i+1)] / 1000. / 1000. / delay_second_list[-(i+1)]) \
                            + (1 - alpha) * throughput_MBps_list[-1]
        throughput_MBps_list.append(exp_avg_throughput / throughput_norm_factor)
        exp_avg_delay = alpha * (delay_second_list[-(i+1)] / delay_norm_factor) \
                        + (1 - alpha) * normed_delay_list[-1]
        normed_delay_list.append(exp_avg_delay)
    # Reverse lists to align with historical ordering
    throughput_MBps_list.reverse()
    normed_delay_list.reverse()
    # Normalizing next chunk size values using max possible chunk size for normalization
    max_next_chunk_size_MB = np.max(next_chunk_bytes_sizes) / 1000. / 1000.
    next_chunk_bytes_MB = [x / 1000. / 1000. / max_next_chunk_size_MB for x in next_chunk_bytes_sizes]
    time_series_states = [
        throughput_MBps_list,
        normed_delay_list,
        next_chunk_bytes_MB,
    ]
    # Return the states
    return {
        "normal_states": normal_states,
        "time_series_states": time_series_states,
    }
```

The best state from GPT-3.5 for the 4G dataset:

```python
import numpy as np
from scipy import stats

def state_func(
    bit_rate_kbps_list,
    buffer_size_second_list,
    delay_second_list,
    video_chunk_size_bytes_list,
    next_chunk_bytes_sizes,
    video_chunk_remain_num,
    total_chunk_num,
    all_bit_rate_kbps,
):
    # Current code remains unchanged for normal states
    normed_last_bit_rate = bit_rate_kbps_list[-1] / float(np.max(all_bit_rate_kbps))
    buffer_norm_factor = 10.
    normed_last_buffer_size = buffer_size_second_list[-1] / buffer_norm_factor
    remaining_chunk_percentage = float(video_chunk_remain_num / total_chunk_num)
    normal_states = [
        [normed_last_bit_rate],
        [normed_last_buffer_size],
        [remaining_chunk_percentage],
    ]
    # Time series state 1: Estimated throughput in near history
    history_window = 8
    throughput_MBps_list = []
    for i in range(history_window):
        history_chunk_size_bytes = video_chunk_size_bytes_list[-(history_window - i)]
        history_delay_second = delay_second_list[-(history_window - i)]
        throughput_MBps_list.append(history_chunk_size_bytes / 1000. / 1000. / history_delay_second)
    # Time series state 2: The normed download time (delay) in near history
    delay_norm_factor = 10.
    normed_delay_list = [x / delay_norm_factor for x in delay_second_list]
    # Time series state 3: Treat next chunk sizes in MB as ts states, too.
    next_chunk_bytes_MB = [x / 1000. / 1000. for x in next_chunk_bytes_sizes]
    # New normal state: Quality of Experience (QoE)
    qoe = (normed_last_bit_rate * normed_last_buffer_size) / np.mean(normed_delay_list)
    # New time series state 4: Variability in throughput and delay
    throughput_variance = np.var(throughput_MBps_list)
    delay_variance = np.var(normed_delay_list)
    # New time series state 5: Trend of throughput and delay
    throughput_trend = stats.linregress(np.arange(history_window), throughput_MBps_list).slope
    delay_trend = stats.linregress(np.arange(history_window), normed_delay_list).slope
    time_series_states = [
        throughput_MBps_list,
        normed_delay_list,
        next_chunk_bytes_MB,
        [qoe],
        [throughput_variance, delay_variance],
        [throughput_trend, delay_trend]
    ]
    return {
        "normal_states": normal_states,
        "time_series_states": time_series_states,
    }
```

The best state from GPT-4 for the 4G dataset:

```python
import numpy as np
# scipy.stats could be used for z-score standardization if we were allowed to import it.

def state_func(
    bit_rate_kbps_list,
    buffer_size_second_list,
    delay_second_list,
    video_chunk_size_bytes_list,
    next_chunk_bytes_sizes,
    video_chunk_remain_num,
    total_chunk_num,
    all_bit_rate_kbps,
):
    # Constants for normalization
    buffer_norm_factor = 10.
    delay_norm_factor = 10.
    max_bit_rate = np.max(all_bit_rate_kbps)
    history_window = min(8, len(bit_rate_kbps_list))  # min to handle shorter lists
    # Normal states
    normed_last_bit_rate = bit_rate_kbps_list[-1] / float(max_bit_rate)
    normed_last_buffer_size = buffer_size_second_list[-1] / buffer_norm_factor
    remaining_chunk_percentage = float(video_chunk_remain_num) / total_chunk_num
    # Bit rate change variance (smoothness metric)
    bit_rate_changes = np.diff(bit_rate_kbps_list[-history_window:]) / max_bit_rate
    smoothness_metric = np.var(bit_rate_changes)
    # Future chunk size ratio
    last_chunk_size_MB = video_chunk_size_bytes_list[-1] / (1000. * 1000.)
    future_chunk_size_ratio = [x / (last_chunk_size_MB * 1000. * 1000.) for x in next_chunk_bytes_sizes]
    # Normal states list
    normal_states = [
        [normed_last_bit_rate],
        [normed_last_buffer_size],
        [remaining_chunk_percentage],
        [smoothness_metric],
    ]
    # Time series states
    # Throughput in near history (standardized)
    throughput_MBps_list = np.array([
        video_chunk_size_bytes_list[-(history_window - i)] / (delay_second_list[-(history_window - i)] * 1000. * 1000.)
        for i in range(history_window)
    ])
    norm_throughput_MBps_list = (throughput_MBps_list - np.mean(throughput_MBps_list)) / np.std(throughput_MBps_list)
    # Download time (delay) in near history (standardized)
    normed_delay_list = (np.array(delay_second_list[-history_window:]) / delay_norm_factor - 1)
    # Exponential Moving Average of throughput and delay
    ema_throughput = np.average(throughput_MBps_list, weights=np.exp(np.arange(history_window)))
    ema_delay = np.average(normed_delay_list, weights=np.exp(np.arange(history_window)))
    # Buffer size trend
    buffer_size_trend = np.diff(buffer_size_second_list[-(history_window + 1):]) / buffer_norm_factor
    # Time series states list
    time_series_states = [
        norm_throughput_MBps_list.tolist(),
        normed_delay_list.tolist(),
        future_chunk_size_ratio,
        [ema_throughput],
        [ema_delay],
        buffer_size_trend.tolist(),
    ]
    # Return the states
    return {
        "normal_states": normal_states,
        "time_series_states": time_series_states,
    }
```


The best state from GPT-3.5 for the 5G dataset:

```python
import numpy as np

def state_func(
    bit_rate_kbps_list,
    buffer_size_second_list,
    delay_second_list,
    video_chunk_size_bytes_list,
    next_chunk_bytes_sizes,
    video_chunk_remain_num,
    total_chunk_num,
    all_bit_rate_kbps,
):
    # existing normal states
    normed_last_bit_rate = bit_rate_kbps_list[-1] / float(np.max(all_bit_rate_kbps))
    buffer_norm_factor = 10.
    normed_last_buffer_size = buffer_size_second_list[-1] / buffer_norm_factor
    remaining_chunk_percentage = float(video_chunk_remain_num / total_chunk_num)
    normal_states = [
        [normed_last_bit_rate],
        [normed_last_buffer_size],
        [remaining_chunk_percentage],
    ]
    # existing time series states
    history_window = 8
    throughput_MBps_list = []
    for i in range(history_window):
        history_chunk_size_bytes = video_chunk_size_bytes_list[-(history_window - i)]
        history_delay_second = delay_second_list[-(history_window - i)]
        throughput_MBps_list.append(history_chunk_size_bytes / 1000. / 1000. / history_delay_second)
    delay_norm_factor = 10.
    normed_delay_list = [x / delay_norm_factor for x in delay_second_list]
    next_chunk_bytes_MB = [x / 1000. / 1000. for x in next_chunk_bytes_sizes]
    time_series_states = [
        throughput_MBps_list,
        normed_delay_list,
        next_chunk_bytes_MB,
    ]
    # new features: variance in throughput
    throughput_variance = np.var(throughput_MBps_list)
    normal_states.append([throughput_variance])
    # new features: prediction of future throughput
    # Utilize a prediction model to estimate future throughput and normalize the result
    predicted_throughput = [0.2, 0.3, 0.4]  # Placeholder for predicted values
    predicted_throughput_normed = [x / float(np.max(throughput_MBps_list)) for x in predicted_throughput]
    normal_states.append(predicted_throughput_normed)
    # new features: statistical features of network conditions
    # Incorporate statistical features such as mean and standard deviation of delay and buffer size
    mean_delay = np.mean(delay_second_list)
    std_buffer_size = np.std(buffer_size_second_list)
    normal_states.append([mean_delay, std_buffer_size])
    return {
        "normal_states": normal_states,
        "time_series_states": time_series_states,
    }
```

The best state from GPT-4 for the 5G dataset:
```python
import numpy as np

def state_func(
    bit_rate_kbps_list,
    buffer_size_second_list,
    delay_second_list,
    video_chunk_size_bytes_list,
    next_chunk_bytes_sizes,
    video_chunk_remain_num,
    total_chunk_num,
    all_bit_rate_kbps,
):
    # Constants for normalization
    buffer_norm_factor = 10.0
    delay_norm_factor = 10.0
    bitrate_norm_factor = np.max(all_bit_rate_kbps)
    size_norm_factor = 1000.0 * 1000.0  # for converting bytes to MB
    # Normalized last bit rate
    normed_last_bit_rate = bit_rate_kbps_list[-1] / bitrate_norm_factor
    # Normalized last buffer size (clipped to max 10 seconds)
    normed_last_buffer_size = np.clip(buffer_size_second_list[-1], 0, buffer_norm_factor) / buffer_norm_factor
    # Percentage of remaining video chunks
    remaining_chunk_percentage = video_chunk_remain_num / total_chunk_num
    # Normal state list
    normal_states = [
        [normed_last_bit_rate],
        [normed_last_buffer_size],
        [remaining_chunk_percentage],
    ]
    # Historical states for bit rates and buffer size differences
    history_window = 8  # for time series state
    normed_bitrate_history = [br / bitrate_norm_factor for br in bit_rate_kbps_list[-history_window:]]
    buffer_size_diffs = np.diff(buffer_size_second_list[-history_window - 1:]) / buffer_norm_factor
    buffer_size_diffs = np.clip(buffer_size_diffs, -1, 1).tolist()  # clip to ensure it stays in range [-1, 1]
    # Estimated throughput in near history normalized
    throughput_MBps_list = [(video_chunk_size_bytes_list[-(history_window - i)] / size_norm_factor) / delay_second_list[-(history_window - i)] for i in range(history_window)]
    # The normed download time (delay) in near history
    normed_delay_list = [(x / delay_norm_factor) for x in delay_second_list[-history_window:]]
    # Throughput stability (variance)
    throughput_variance = np.var(throughput_MBps_list) / np.var(all_bit_rate_kbps)
    # Sizes for the next chunk normalized
    next_chunk_sizes_norm = [size / size_norm_factor for size in next_chunk_bytes_sizes]
    # Time series states list
    time_series_states = [
        normed_bitrate_history,
        buffer_size_diffs,
        throughput_MBps_list,
        normed_delay_list,
        next_chunk_sizes_norm,
        [throughput_variance],  # included as a single-element list for consistency
    ]
    return {
        "normal_states": normal_states,
        "time_series_states": time_series_states,
    }
```


### Top Performing Network Architectures


The best network architecture from GPT-3.5 for the FCC dataset:

```python
import tensorflow.compat.v1 as tf
import tflearn

def network_func(normal_input_list, ts_input_list, action_dim):
    with tf.variable_scope('actor'):
        normal_features = [
            tflearn.fully_connected(normal_input, 256, activation='leaky_relu')
            for normal_input in normal_input_list
        ]
        ts_features = [
            tflearn.flatten(tflearn.conv_1d(
                tf.expand_dims(ts_input, axis=1), 
                256, 1, activation='leaky_relu'
            ))
            for ts_input in ts_input_list
        ]
        merged_features = tflearn.merge(normal_features + ts_features, "concat")
        pi_features = tflearn.fully_connected(merged_features, 256, activation='leaky_relu')
        pi = tflearn.fully_connected(pi_features, action_dim, activation='softmax')

    with tf.variable_scope('critic'):
        normal_features = [
            tflearn.fully_connected(normal_input, 256, activation='leaky_relu')
            for normal_input in normal_input_list
        ]
        ts_features = [
            tflearn.flatten(tflearn.conv_1d(
                tf.expand_dims(ts_input, axis=1), 
                256, 1, activation='leaky_relu'
            ))
            for ts_input in ts_input_list
        ]
        merged_features = tflearn.merge(normal_features + ts_features, "concat")
        value_features = tflearn.fully_connected(merged_features, 256, activation='leaky_relu')
        value = tflearn.fully_connected(value_features, 1, activation='linear')

    return pi, value
```

The best network architecture from GPT-3.5 for the Starlink dataset:

```python
import tensorflow.compat.v1 as tf
import tflearn

def network_func(normal_input_list, ts_input_list, action_dim):
    with tf.variable_scope('actor'):
        normal_features = [
            tflearn.fully_connected(normal_input, 256, activation='relu')
            for normal_input in normal_input_list
        ]
        ts_features = [
            tflearn.simple_rnn(tf.expand_dims(ts_input, axis=2), 128, return_seq=False)
            for ts_input in ts_input_list
        ]
        merged_features = tflearn.merge(normal_features + ts_features, "concat")
        pi_features = tflearn.fully_connected(merged_features, 256, activation='relu')
        pi = tflearn.fully_connected(pi_features, action_dim, activation='softmax')

    with tf.variable_scope('critic'):
        normal_features = [
            tflearn.fully_connected(normal_input, 256, activation='relu')
            for normal_input in normal_input_list
        ]
        ts_features = [
            tflearn.simple_rnn(tf.expand_dims(ts_input, axis=2), 128, return_seq=False)
            for ts_input in ts_input_list
        ]
        merged_features = tflearn.merge(normal_features + ts_features, "concat")
        value_features = tflearn.fully_connected(merged_features, 256, activation='relu')
        value = tflearn.fully_connected(value_features, 1, activation='linear')

    return pi, value
```

The best network architecture from GPT-3.5 for the 4G dataset:
```python
import tensorflow.compat.v1 as tf
import tflearn

def network_func(normal_input_list, ts_input_list, action_dim):
    with tf.variable_scope('actor'):
        # Process normal input features using fully connected layers
        normal_fc = tf.concat(normal_input_list, axis=1)
        normal_fc = tflearn.fully_connected(normal_fc, 256, activation='relu')
        
        # Process time series input features using LSTM cells
        ts_input = tf.concat(ts_input_list, axis=1)
        ts_input = tf.expand_dims(ts_input, -1)  # Add an extra dimension for input to LSTM
        lstm_cell = tf.nn.rnn_cell.LSTMCell(128)
        lstm_output, _ = tf.nn.dynamic_rnn(lstm_cell, ts_input, dtype=tf.float32)
        ts_lstm = tflearn.flatten(lstm_output)
        
        # Merge processed features and pass through actor network
        merged_features = tf.concat([normal_fc, ts_lstm], axis=1)
        pi_features = tflearn.fully_connected(merged_features, 128, activation='relu')
        pi = tflearn.fully_connected(pi_features, action_dim, activation='softmax')

    with tf.variable_scope('critic'):
        # Process normal input features using fully connected layers
        normal_fc_critic = tf.concat(normal_input_list, axis=1)
        normal_fc_critic = tflearn.fully_connected(normal_fc_critic, 256, activation='relu')
        
        # Process time series input features using LSTM cells
        lstm_output_critic, _ = tf.nn.dynamic_rnn(lstm_cell, ts_input, dtype=tf.float32)  # Reuse LSTM cell
        ts_lstm_critic = tflearn.flatten(lstm_output_critic)
        
        # Merge processed features and pass through critic network
        merged_features_critic = tf.concat([normal_fc_critic, ts_lstm_critic], axis=1)
        value_features = tflearn.fully_connected(merged_features_critic, 128, activation='relu')
        value = tflearn.fully_connected(value_features, 1, activation='linear')

    return pi, value
```

The best network architecture from GPT-3.5 for the 5G dataset:
```python
import tensorflow.compat.v1 as tf
import tflearn

def network_func(normal_input_list, ts_input_list, action_dim):
    with tf.variable_scope('actor_critic'):
        normal_features = [
            tflearn.fully_connected(normal_input, 256, activation='relu')
            for normal_input in normal_input_list
        ]
        ts_features = [
            tflearn.flatten(tflearn.conv_1d(
                tf.expand_dims(ts_input, axis=1), 
                256, 1, activation='relu'
            ))
            for ts_input in ts_input_list
        ]
        merged_features = tflearn.merge(normal_features + ts_features, "concat")
        hidden_layer = tflearn.fully_connected(merged_features, 512, activation='relu')
        
        pi = tflearn.fully_connected(hidden_layer, action_dim, activation='softmax')
        value = tflearn.fully_connected(hidden_layer, 1, activation='linear')

    return pi, value
```

