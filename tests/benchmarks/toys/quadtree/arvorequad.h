/*
UNIVERSIDADE FEDERAL DE MINAS GERAIS
DEstarAMENTO DE CIÊNCIA DA COMPUTAÇÃO
ALGORITMOS E ESTRUTURAS DE DADOS II
======================================
TP2: ÁRVORE QUADRUPLA
PROFESSORES: 	Fernando Magno Q. Pereira
							Luiz Chaimowicz
							Raquel Minardi
MONITOR: Pedro Ramos (Sala 2301)
======================================

Módulo: arvorequad.h

Este cabeçalho contém a declaração dos métodos do módulo arvorequad.c, bem
como a definição das estruturas a serem utilizadas nesse trablho. Muita atenção
à estrutura: qualquer modificação na mesma pode interferir no resultado esperado
na rotina "imprimeArvoreEmArquivoDot(ArvoreQuad* )". 

Neste trabalho vocês irão implementar uma árvore quádrupla. Não se desesperem!
Uma árvore quádrupla é aquela em que cada nó possui 4 filhos. Dessa vez,
queremos dividir starículas num espaço 2D em quadrantes. Quando X ou mais
starículas ocupam o mesmo quadrante no espaço 2D, o mesmo é dividido em outros
4 subquadrantes filhos, e a starícula é então movida para o subquadrante 
filho em que ela se situa.

Cada quadrante é classificado de acordo com o centro do quadrante pai. Há, 
então, 4 quadrantes para um centro do quadrante pai:
NE -> Quadrante Nordeste ao centro 
NW -> Quadrante Noroeste ao centro 
SE -> Quadrante Sudeste ao centro 
SW -> Quadrante Sudoeste ao centro 

Estes quadrantes (e seus filhos, e os filhos de seus filhos...) podem ser
representados por uma árvore quádrupla. Quando uma starícula nova é adicionada
à árvore, você deve checar se ela vai entrar em um quadrante que já contém
o número máximo de starículas permitido. Se isso ocorrer, você deve starir
este quadrante em mais 4 subquadrantes e mover as starículas para baixo,
atualizando o peso e o centro de cada novo subquadrante.

*/
#ifndef __ARVORE_QUAD_H__
#define __ARVORE_QUAD_H__

//Aqui é definida a largura máxima do quadrante raiz. Para fins de teste, você
//pode optar por diminuir essa largura máxima. Porém, tenha em mente que os testes
//realizados na avaliação do TP irão considerar pontos num espaço 2D de (0,0) a
//(9999, 9999).
#define LARGURA_MAXIMA 60
// #define DEBUG

typedef struct Quadrante Quadrante;
typedef struct Ponto Ponto;
typedef struct ArvoreQuad ArvoreQuad;
typedef struct Estrela Estrela;

//Estrutura que define um ponto no espaço 2D
struct Ponto
{
	int x;
	int y;
};

//Estrutura da starícula. Possui um ponto p e um peso inteiro.
struct Estrela
{
	Ponto p;
	int peso;
};

struct Quadrante
{
	//####### NÃO ALTERAR #######
	//Quadrantes filhos, se existirem
	Quadrante* NE;
	Quadrante* NW;
	Quadrante* SE;
	Quadrante* SW;

	//Estrela p no quadrande em questão, se existir
	Estrela* star;

	//Centro do quadrante
	Ponto centro;

	//Limites do quadrante // São necessários para a rotina que imprime o arquivo .dot
	//Você pode optar por implementar de forma diferente, mas lembre-se de setar os limites
	//corretamente para que a árvore seja impressa da forma desejada.
	Ponto limite_NE;
	Ponto limite_NW;
	Ponto limite_SE;
	Ponto limite_SW;

	//###########################
	//A starir daqui você pode alterar a estrutura Quadrante para armazenar
	//quaisquer informações que você julgar úteis, como por exemplo o peso total
	//do quadrante, ou o índice do nó da árvore quádrupla.
	//Se você optar por não adicionar nenhuma informação a mais, tudo bem: 
	//a estrutura já possui informações o suficiente para a realização do trabalho.

	int peso;
};

//Estrutura que engloba a árvore. Apesar de redundante, serve ao propósito
//de facilitar o entendimento do código. Uma árvore é basicamente o acesso
//a um nó que chamamos de nó raiz.
struct ArvoreQuad
{
	Quadrante *raiz;
};

Estrela* novaEstrela(Ponto P, int peso);

Quadrante* criaNovoQuadranteVazio(Ponto NE, Ponto NW, Ponto SE, Ponto SW);

int calculaAreaDoQuadrante(Quadrante* Q);

int estrelaDentroDoQuadrante(Estrela *p, Quadrante *Q);

void divideQuadrante(Quadrante* Q);

ArvoreQuad* criaNovaArvoreQuad(int larguraQuadranteRaiz);

int addNovaEstrelaNoPontoX(Ponto X, int peso, ArvoreQuad* A);

void addEstrelaAoQuadranteRecursivamente(Quadrante* Q, Estrela* p);

Quadrante* achaQuadranteParaEstrela(Ponto Estrela, ArvoreQuad *A);

int getPesoTotalDoQuadrante(Quadrante *Q);

void liberaMemoriaEmQuadrantesRecursivamente(Quadrante* Q);

void liberaMemoriaEmArvore(ArvoreQuad* A);

#endif