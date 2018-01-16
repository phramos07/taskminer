#include "arvorequad.h"
#include <stdlib.h>
#include <stdio.h>
// #define DEBUG

void imprimePesoDeTodosOsQuadrantes(Quadrante* Q, int index)
{
	if (Q == NULL)
		return;
	imprimePesoDeTodosOsQuadrantes(Q->NE, index*4 + 1);
	imprimePesoDeTodosOsQuadrantes(Q->NW, index*4 + 2);
	imprimePesoDeTodosOsQuadrantes(Q->SW, index*4 + 3);
	imprimePesoDeTodosOsQuadrantes(Q->SE, index*4 + 4);
	Q->peso = getPesoTotalDoQuadrante(Q);
	printf("(%d, %d) %d\n", Q->centro.x, Q->centro.y, Q->peso);
}

int main(int argc, char const *argv[])
{
	int maxLargura, x, y, peso, pesototal=0;
	unsigned numPontos;
	scanf("%d", &maxLargura);
	scanf("%d", &numPontos);
	ArvoreQuad* A = criaNovaArvoreQuad(maxLargura);
	while (numPontos > 0)
	{
		x = rand()%maxLargura;
		y = rand()%maxLargura;
		peso = rand()%100;
		// scanf("%d %d %d", &x, &y, &peso);
		Ponto P;
		P.x = x;
		P.y = y;
		if (addNovaEstrelaNoPontoX(P, peso, A))
			pesototal += peso;
		numPontos--;
	}

	#ifdef DEBUG
		imprimePesoDeTodosOsQuadrantes(A->raiz, 0);
		if (pesototal != getPesoTotalDoQuadrante(A->raiz))
			printf("PESO INCORRETO\n");
	#endif

	liberaMemoriaEmArvore(A);
	return 0;
}

Estrela* novaEstrela(Ponto P, int peso)
{
	Estrela* star = (Estrela*) malloc(sizeof(Estrela));
	star->p = P;
	star->peso = peso;

	return star;
}

Quadrante* criaNovoQuadranteVazio(Ponto NE, Ponto NW, Ponto SE, Ponto SW)
{
	Quadrante* Q = (Quadrante*) malloc(sizeof(Quadrante));
	Q->NE = NULL;
	Q->NW = NULL;
	Q->SE = NULL;
	Q->SW = NULL;	
	Q->star = NULL;

	Q->limite_NE = NE;
	Q->limite_NW = NW;
	Q->limite_SE = SE;
	Q->limite_SW = SW;

	//Derivar o centro do quadrante
	int x_centro = (Q->limite_SE.x - Q->limite_SW.x)/2 + Q->limite_SW.x;
	int y_centro = (Q->limite_NE.y - Q->limite_SE.y)/2 + Q->limite_SE.y;
	Ponto centro;
	centro.x = x_centro;
	centro.y = y_centro;
	Q->centro = centro;

	return Q;
}

ArvoreQuad* criaNovaArvoreQuad(int larguraQuadranteRaiz)
{
	ArvoreQuad* A = (ArvoreQuad*) malloc(sizeof(ArvoreQuad));
	Ponto centro, NE, NW, SE, SW;
	centro.x = larguraQuadranteRaiz/2;
	centro.y = larguraQuadranteRaiz/2;

	SW.x = 0;
	SW.y = 0;

	NW.x = 0;
	NW.y = larguraQuadranteRaiz;

	NE.x = larguraQuadranteRaiz;
	NE.y = larguraQuadranteRaiz;

	SE.x = larguraQuadranteRaiz;
	SE.y = 0;

	A->raiz = criaNovoQuadranteVazio(NE, NW, SE, SW);

	return A;
}

int addNovaEstrelaNoPontoX(Ponto P, int peso, ArvoreQuad* A)
{
	Estrela* nova = novaEstrela(P, peso);
	if (estrelaDentroDoQuadrante(nova, A->raiz))
	{
		addEstrelaAoQuadranteRecursivamente(A->raiz, nova);
		return 1;
	}
	free(nova);
	return 0;
}

int estrelaDentroDoQuadrante(Estrela *star, Quadrante *Q)
{
	Ponto P = star->p;
	//veririca se está dentro do quadrante
	if ((P.x >= Q->limite_SW.x && P.x < Q->limite_SE.x) && 
		(P.y >= Q->limite_SW.y && P.y < Q->limite_NW.y))
		return 1;

	return 0;
}

int calculaAreaDoQuadrante(Quadrante* Q)
{
	int y = Q->limite_NE.y - Q->limite_SE.y;
	int x = Q->limite_SE.x - Q->limite_SW.x;

	return y*x;
}

void divideQuadrante(Quadrante* Q)
{
	//TODO: não dividir se tiver área menor ou igual a 1.
	if (calculaAreaDoQuadrante(Q) <= 1)
	{
		#ifdef DEBUG
			printf("Quadrante indivisível.\n");
		#endif
		return;
	}

	//Define os limites dos novos quadrantes
	Ponto norte, sul, leste, oeste;
	norte.x = (Q->limite_NE.x - Q->limite_NW.x)/2 + Q->limite_NW.x;
	norte.y = Q->limite_NE.y;

	sul.x = (Q->limite_SE.x - Q->limite_SW.x)/2 + Q->limite_SW.x;
	sul.y = Q->limite_SE.y;

	leste.x = Q->limite_NE.x;
	leste.y = (Q->limite_NE.y - Q->limite_SE.y)/2 + Q->limite_SE.y;

	oeste.x = Q->limite_NW.x;
	oeste.y = (Q->limite_NW.y - Q->limite_SW.y)/2 + Q->limite_SW.y;

	Q->NE = criaNovoQuadranteVazio(Q->limite_NE, norte, leste, Q->centro);
	Q->NW = criaNovoQuadranteVazio(norte, Q->limite_NW, Q->centro, oeste);
	Q->SE = criaNovoQuadranteVazio(leste, Q->centro, Q->limite_SE, sul);
	Q->SW = criaNovoQuadranteVazio(Q->centro, oeste, sul, Q->limite_SW);

}

void addEstrelaAoQuadranteRecursivamente(Quadrante* Q, Estrela* star)
{
	//ponto da Estrela p deve estar dentro do quadrante Q. Se não estiver, saia.
	//Faça uma função para verificar se o ponto está dentro do quadrante.
	if (!estrelaDentroDoQuadrante(star, Q))
		return;
	
	#ifdef DEBUG
		printf("Tentando adicionar a Estrela no ponto: (%d, %d)\n", star->p.x, star->p.y);
	#endif

	//Se o quadrante não pode mais ser dividido, retorne.
	if (calculaAreaDoQuadrante(Q) <= 1)
	{
		if (Q->star == NULL && Q->NE == NULL)
		{
			//Se está vazio e não tem filhos, adiciona a starícula e saia.
			Q->star = star;
			return;
		}
		else
		{
			Q->star->peso += star->peso;
		}
		return;
	}



	//Verifica se Q está vazio e não tem filhos
	if (Q->star == NULL && Q->NE == NULL)
	{
		//Se está vazio e não tem filhos, adiciona a starícula e saia.
		Q->star = star;
		return;
	}
	else
	{
		//Verifica se Q é dividido em subquadrantes
		if (Q->NE != NULL) //Qualquer filho não-nulo significa que Q é dividido
		{
			//tenta adicionar a starícula em todos os filhos
			addEstrelaAoQuadranteRecursivamente(Q->NE, star);
			addEstrelaAoQuadranteRecursivamente(Q->NW, star);
			addEstrelaAoQuadranteRecursivamente(Q->SE, star);
			addEstrelaAoQuadranteRecursivamente(Q->SW, star);
		}
		else //se não for dividido, divide Q em subquadrantes
		{
			Estrela* aux = Q->star;

			//esvazia o quadrante atual
			Q->star = NULL;

			//Divide Q em subquadrantes
			divideQuadrante(Q);

			//Passa a starícula em Q para um quadrante abaixo
			addEstrelaAoQuadranteRecursivamente(Q->NE, aux);
			addEstrelaAoQuadranteRecursivamente(Q->NW, aux);
			addEstrelaAoQuadranteRecursivamente(Q->SE, aux);
			addEstrelaAoQuadranteRecursivamente(Q->SW, aux);

			//tenta adicionar a starícula em todos os filhos
			addEstrelaAoQuadranteRecursivamente(Q->NE, star);
			addEstrelaAoQuadranteRecursivamente(Q->NW, star);
			addEstrelaAoQuadranteRecursivamente(Q->SE, star);
			addEstrelaAoQuadranteRecursivamente(Q->SW, star);
		}
	}
}

void liberaMemoriaEmArvore(ArvoreQuad* A)
{
	liberaMemoriaEmQuadrantesRecursivamente(A->raiz);
	free(A);
}

void liberaMemoriaEmQuadrantesRecursivamente(Quadrante* Q)
{
	if (Q == NULL)
		return;
	liberaMemoriaEmQuadrantesRecursivamente(Q->NE);
	liberaMemoriaEmQuadrantesRecursivamente(Q->NW);
	liberaMemoriaEmQuadrantesRecursivamente(Q->SE);
	liberaMemoriaEmQuadrantesRecursivamente(Q->SW);
	if (Q->star != NULL)
	{
		free(Q->star);
	}
	if (Q != NULL)
		free(Q);

	return;
}

Quadrante* achaQuadranteParaEstrela(Ponto Estrela, ArvoreQuad *A)
{ 
	return NULL; 
}

int getPesoTotalDoQuadrante(Quadrante *Q)
{
	if (Q == NULL)
		return 0;

	int peso;
	if (Q->star == NULL)
		peso = 0;
	else
		peso = Q->star->peso;

	return 	getPesoTotalDoQuadrante(Q->NE) + 
					getPesoTotalDoQuadrante(Q->SE) +
					getPesoTotalDoQuadrante(Q->NW) +
					getPesoTotalDoQuadrante(Q->SW) +
					peso;
}

