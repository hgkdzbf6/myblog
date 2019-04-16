---
layout: post
title:  "tensorflow入门"
date:   2019-03-22 17:40:32 +0800
categories: Python
---

## 为什么写这篇文章？

因为tensorflow不会啊！

## 一些记录：

### 图的保存：

{% codeblock lang:python %}
writer = tf.summary.FileWriter("./log",sess.graph)
writer.close()
{% endcodeblock %}

### checkpoint的保存：

{% codeblock lang:python %}

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
saver = tf.train.Saver()
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())     # 初始化计算图中的变量
{% endcodeblock %}

注意后面还要在循环当中保存：
{% codeblock lang:python %}
saver.save(sess, checkpoint_dir +'model.ckpt', global_step=step+1)
{% endcodeblock %}

### 学习lstm

参考[这里](https://blog.csdn.net/xierhacker/article/details/78772560)