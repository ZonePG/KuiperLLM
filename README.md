# 大模型推理框架 KuiperLLM

## LLama3.2 推理

- 以 meta-llama/Llama-3.2-1B 为例，huggingface 上下载模型：
```shell
export HF_ENDPOINT=https://hf-mirror.com
pip3 install huggingface-cli
huggingface-cli download --resume-download meta-llama/Llama-3.2-1B --local-dir meta-llama/Llama-3.2-1B --local-dir-use-symlinks False
```
- 导出模型：
```shell
python3 tools/export.py Llama-3.2-1B.bin --hf=meta-llama/Llama-3.2-1B
```
- 编译：
```shell
mkdir build 
cd build
# 开启 USE_CPM 选项，自动下载第三方依赖，前提是需要网络畅通
cmake -DUSE_CPM=ON -DLLAMA3_SUPPORT=ON .. 
make -j16
```
- 运行：
```shell
./build/demo/llama_infer Llama-3.2-1B.bin meta-llama/Llama-3.2-1B/tokenizer.json
# 和 huggingface 推理的结果进行对比
python3 hf_infer/llama3_infer.py
```

## Qwen2.5 推理

- 以 Qwen2.5-0.5B 为例，huggingface 上下载模型：
```shell
export HF_ENDPOINT=https://hf-mirror.com
pip3 install huggingface-cli
huggingface-cli download --resume-download Qwen/Qwen2.5-0.5B --local-dir Qwen/Qwen2.5-0.5B --local-dir-use-symlinks False
```
- 导出模型：
```shell
python3 tools/export_qwen2.py Qwen2.5-0.5B.bin --hf=Qwen/Qwen2.5-0.5B
```
- 编译：
```shell
mkdir build 
cd build
# 开启 USE_CPM 选项，自动下载第三方依赖，前提是需要网络畅通
cmake -DUSE_CPM=ON -DQWEN2_SUPPORT=ON .. 
make -j16
```
- 运行：
```shell
./build/demo/qwen_infer Qwen2.5-0.5B.bin Qwen/Qwen2.5-0.5B/tokenizer.json
# 和 huggingface 推理的结果进行对比
python3 hf_infer/qwen2_infer.py
```

## 实现细节

### 模型初始化 init()

- gen_model_from_file(): 从文件中读取模型
  - create_encode_layer(): 读取 tokenizer path，初始化 google sentencepiece tokenizer 模型。
  - read_model_file(): 读取 model path 模型文件，获取模型配置信息，并通过 mmap 将模型文件数据地址其映射到内存中。（用 mmap 的好处是将文件直接映射到进程的地址空间读写操作更高效、按需加载可以减少内存占用）
  - create_layers()
    - create_param_layer(): 创建有权重参数的层，并初始化参数。包括：
      - embedding: 0 - dim * std::abs(config_->vocab_size_)
      - attention weights: attention_norm, attention.wq, attention.wk, attention.wv, attention.wo
      - ffn weights: ffn_norm, feed_forward.w1, feed_forward.w2, feed_forward.w3
      - final rmsnorm
      - freq_cis (skip)
      - final classifier weights
    - create_nonparam_layers(): 创建没有权重参数的层，包括：RoPELayer、MultiHeadAttention、VecAddLayer、SwiGLULayer
- init_mem(): 初始化模型输入输出空间的内存资源
  - llama_layers_->to_cuda(cuda_config_)：将模型参数拷贝到 GPU 内存中 
  - input_tokens、input_embeddings、sin_cache 和 cos_cache 等张量的创建: 为模型计算所需的数据分配内存，包括输入、位置编码等。
  - insert_buffer 方法的调用: 将创建的张量插入到模型的缓存区管理器中，同时复用内存空间。
- kernel::sin_cos_cache_calc: 计算位置编码的 sin 和 cos 值
- sampler::ArgmaxSampler: 使用 argmax 进行贪婪采样

### 生成 token generate()

- tokenizer encode(): 通过 sentencepiece tokenizer 对输入文本进行编码，生成 token id
- embedding(): 通过 token id 获取 token 的 embedding 向量
- model.fill_input(): 将 embedding 向量填充到模型的输入张量中
- model.predict(input, pos_tensor, is_prompt, next): 预测下一个 token 的 id
  - forward(): 遍历每一层，接受输入张量，计算输出张量
    - 每一层的前向
      - attention_rms(): 通过 kernel::get_rmsnorm_kernel() 计算 rmsnorm 层的输出
      - attention_qkv()
        - 从 buffer 中获取要写入的 qeury，从 buffer KV cache 中获取要写入的 key 和 value，分别计算通过矩阵乘法 kernel::get_matmul_kernel() 计算 q、k、v
        - 通过 qeury, key, pos_tensor, sin_cache, cos_cache 为 query 和 key 添加旋转位置编码
      - attention_mha(): 执行多头注意力机制的前向计算
        - 从 buffer 中取出 kv cache，和 qeury
        - 通过 kernel::get_mha_kernel() 计算多头注意力机制的输出
        - 通过 wo 线性层计算 attn_output
      - feed_forward(): 残差连接和前馈神经网络的前向计算
        - 上一层输入和输出的残差连接
        - rmsnorm
        - w1 w3 计算 SwiGLU -> w2
        - 上一层输入经过 rmsnorm 和 feed_forward 计算输出的残差链接
    - cls_logits(): 最后一层的输出通过 classifier 线性层计算输出概率分布
      - rmsnorm
      - cls_layer 计算输出 logits (vocab_size)
    - post_process(): 对输出 logits 进行后处理
      - 如果是 prefill 阶段，则不贪婪采样
      - 如果是 decode 阶段，则使用 argmax 进行贪婪采样
