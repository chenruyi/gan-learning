# Generative Adversarial Nets
Here are notes on the paper about Generative Adversarial Nets ( abbr. GAN), the paper is:[Generative Adversarial Nets](#http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

## Main Point
The GAN idea is interesting and naive in computer science.
GAN is a generating model.  GAN is a framework for teaching a DL model to capture the training data's distribution so we can generate new data from that same distribution.

### What
<!-- reference:  
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#what-is-a-gan
https://www.tensorflow.org/beta/tutorials/generative/dcgan#what_are_gans
-->
GAN contains two distinct models: a generator and a discriminator. We consider about the generator, which will produce a new data, such as, image, voice, etc. The generator job is to 'create' a image that look like training data, while the discriminator learns to classify the real and create image. During training the generator trying to 'fake' the discriminator by generating better and better fakes, while the discriminator is going to tell the real and fake data apart. Finally this game will reach the equilibrium. The generator can provide prefect fake data that look real and the discriminator can't distinguish data whether is from the real or the generator and always guess 50% true and 50% false.

The idea is influenced by behaviorism. The generator and the discriminator is what they do, the reason behind is unknown. The data's distribution is learned by the struggle between the generator and discriminator. And the success is that the generator and discriminator behave as we want.

Reference: 
- [pytorch.org](#https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#what-is-a-gan)
- [tensorflow.org](#https://www.tensorflow.org/beta/tutorials/generative/dcgan#what_are_gans)

### Contribution
this first paper introduce GAN by Ian Goodfellow in 2016.

[generating model without Markov chains and no inference. So GAN can represent sharp, even degenerate distributions. While methods based on Markov chains require that the distribution be somewhat blurry in order for the chains to be able to mix between modes](#Reference:GAN_paper_6_Advantages_and_disadvantages)

the generating model without lots of special domain knowledge.
can apply in lots of domain, and lots of kinds problem, including image reproduce, image super resolution, voice reproduce, text reproduce, 3D model generating, and so on.

behaviorism influence.

## Notes

### Architecture
the GAN architecture is two part: Generator(G) and Discriminator(D). 

define notation, using MINIST image as an example:

$x$ -> Image, real data. e.g. 28 $*$ 28 1bit in MINIST dataset.

$G$ -> Generator, a network, maps the $z$ tp data-space. input: $z$, output($G(z)$): image, fake data, like $x$.

$z$ -> a latent space vector sampled from stand normal distribution. Usually random number.

$D$ -> Discriminator, a network, input: $x$ or $G(z)$(the Generator output), output($D(x)$ or $D(G(z))$):classify whether the input is $x$(real) or $G(z)$(fake), usually the probability of real or 0,1.

![](images/批注&#32;2019-06-28&#32;164240.png)
<!-- reference: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#what-is-a-gan -->
The goal of $G$ is to estimate the distribution that the training data comes from $p_{data}$, so it can generate fake samples from that estimated distribution($p_g$)。

So, $D(G(z))$ is the probability that the output of the generator $G$ is a real images. As described in [Goodfellow's paper](#http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf), $D$ and $G$ play a minimax game in which $D$ tries to maximize the probability it correctly classifies reals and fakes($logD(x)$), and G tries to minimize the probability that D will predict its outputs are fake($log(1-D(G(x)))$).
From the paper, the GAN loss function is
$\underset{G}{\text{min}} \underset{D}{\text{max}}V(D,G) = \mathbb{E}_{x\sim p_{data}(x)}\big[logD(x)\big] + \mathbb{E}_{z\sim p_{z}(z)}\big[log(1-D(G(z)))\big]$

In theory, the solution to this minimax game is where $P_g=P_{data}$, and the discriminator guesses randomly if the inputs are real or fake. However, the convergence theory of GANs is still being researched and in reality models do not always train to this point.

Reference: 
- [pytorch.org](#https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#what-is-a-gan)
   
### loss
the two
#### formula
<!-- TODO -->
Setting: 
- nonparametric. represent a model with infinite capacity by studying convergence in the space of probability density functions.[paper 4 Theoretical Results](#paper_4_Theroretical_Results)
- consider discriminator D for any given generator G.
    There are D and G. And the D loss is simple and clear(D is to solve two classification questions).

Fixed G (or given any generator G). The discriminator D is to maximize the $V(G,D)$.

$\begin{aligned}
    V(G,D) &= \int_{x}p_{data}(x)log(D(x))dx + \int_{z} p_{z}(z)log(1-D(g(z))dz\\&=\int_{x}p_{data}(x)log(D(x)) + p_{g}(x)log(1-D(x)) dx
\end{aligned}$

Here according to the function $f(y)=a log(y) + b log(1-y)$, this achieves its maximum in $y=\frac{a}{a+b}$, when $y \in [0,1]$.
Because the $x$ can normalize to $[0,1]$, so the optimal discriminator D is
$D^*_G(x)=\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}$
That's equal to paper [Proposition 1](#Proposition_1). Now the [Proposition 1](#Proposition_1) is proved.
###### Proposition 1
*For G ﬁxed, the optimal discriminator D is
$D^*_G(x)=\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}$*

Also the training objective for D can be interpreted as maximizing the log-likelihood for estimating the conditional probability $P(Y=y|x)$, where $Y$ indicates whether $x$(input of D) comes from $p_{data}$ (with $y=1$) or from $p_g$ (with $y=0$). 
Using the [Proposition 1](#Proposition_1), the goal(the GAN loss function) can now be reformulated as:
$\begin{aligned}
    C(G) &= \underset{G}{max}V(G,D)\\
        &= \mathbb{E}_{x\sim p_{data}}[logD^*_G(x)]+\mathbb{E}_{z\sim p_z}[log(1-D^*_G(G(z)))]\\
        &=\mathbb{E}_{x\sim p_{data}}[logD^*_G(x)]+\mathbb{E}_{x\sim p_g}[log(1-D^*_G(x))]\\
        &= \mathbb{E}_{x\sim p_{data}}[log{\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}}]+\mathbb{E}_{x\sim p_g}[log{\frac{p_g(x)}{p_{data}(x)+p_g(x)}}]
\end{aligned}$


###### Theorem 1
*The global minimum of the virtual training criterion C(G) is achieved if and only if $p_g=p_{data}$. At that point, C(G) achieves the values -log4.*
Thinking about the the generator G. Now we finish the inner question $\underset{D}{max}$. Considering the $\underset{G}{min}$. However let's begins in an easy way.First we prove the equal.
Inspecting the define equation $D^*_G(x)$ at $p_g=p_{data}$, $D^*_G(x)=\frac{1}{2}$. and we will find $C(G)=log\frac{1}{2}+log\frac{1}{2}=-log4$. So the equal proved.
Now proved the $C(G)$ reach the minimum.

$\begin{aligned}
    &\mathbb{E}_{x\sim p_{data}}[-log2]+\mathbb{E}_{x\sim p_{data}}[-log2]=-log4 =>\\
    &log4+\mathbb{E}_{x\sim p_{data}}[-log2]+\mathbb{E}_{x\sim p_{data}}[-log2]  = 0 
\end{aligned}$
So,
$\begin{aligned}
    C(G)&=\mathbb{E}_{x\sim p_{data}}[log{\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}}]+\mathbb{E}_{x\sim p_g}[log{\frac{p_g(x)}{p_{data}(x)+p_g(x)}}]-0\\
        &=-log4+\mathbb{E}_{x\sim p_{data}}[log{\frac{p_{data}(x)}{\frac{p_{data}(x)+p_g(x)}{2}}}]+\mathbb{E}_{x\sim p_g}[log{\frac{p_g(x)}{\frac{p_{data}(x)+p_g(x)}{2}}}]\\
        &=-log4+KL \left( p_{data}||\frac{p_{data}+p_g}{2} \right) + KL\left( p_{g}||\frac{p_{data}+p_g}{2}\right)\\&=-log4+2\cdot JSD\left(p_{data}||p_g \right)
\end{aligned}$
So the Jensen-Shannon divergence(JSD) between two distributions is always not-negative and zero only when they are equal, so the prove only $p_g=p_{data}$, the $C(G)$ reach the global minimum. The $p_g=p_{data}$ i.e., the generative reproduce the fake data, which looks like the real.


### train function

Algorithm 1 Minibatch SGD training.

![](images/批注&#32;2019-06-29&#32;110706.png)


the main point in training is that:
- discriminator D train k steps and generator G train one step.
- discriminator gradient ascending(to get the loss maximum).
- generator gradient descending(to get the loss minimum).

###### Proposition 2
*If G and D have enough capacity, and at each of Algorithm 1, the discriminator is allowed to reach its optimum given G, and $p_g$ is updated so as to improve the criterion*
$\mathbb{E}_{x\sim p_{data}}[logD^*_G(x)]+\mathbb{E}_{x\sim p_g}[log(1-D^*_G(x))] $
*then $p_g$ converges to p_{data}*

Now we have proved the GAN loss function real work, but the training methods work as we design? This proposition 2 proved that. The training methods main point is how to train the generator G. The D is to solve two classification question.
<!-- Reference: paper 4.2  Convergence of Algorithm1  -->
Consider $V(G,D)=U(p_g,D)$ as a function of $p_g$ as done in the above proof. $U(p_g,D) is convex in p_g$. The subderivatives of a supremum of convex functions includes the derivative of the function at the point where the maximum is attained. In others words, this is equivalent to computing a gradient descent update for $p_g$ at the optimal D given the corresponding $G$. $sup_D U(p_g,D)$ is convex in $p_g$ with a unique global optima as proven in Thm 1, therefore with sufficiently small updates of $p_g$,$p_g$ converges to $p_x$, concluding the proof.

In practice, adversarial nets represent a limited family of $p_g$distributions via the function $G(z;\theta_g)$, and we optimize $theta_g$ rather than $p_g$ itself. Using a multilayer perceptron to define G introduces multiple critical points in parameter space. However, the excellent performance or multilayer perceptrons in practice suggests that they are reasonable model to use despite their lack of theoretical guarantees.
Reference: [paper 4.2 Convergence of Algorithm1](#paper_4.2_Convergence_of_Algorithm1)


Obviously, this proof need the network trained as we want and the SGD methods really work well.



## Critique

training is difficulty.

## Appendix

the code are minist-gan.ipynb


## Reference
[Generative Adversarial Nets](#http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)


## TODO
- [x] GAN idea
- [x] GAN architecture
- [x] GAN loss
- [x] GAN train function
- [ ] GAN code recurrent
