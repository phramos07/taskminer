#include <omp.h>
#ifndef taskminerutils
#define taskminerutils
static int taskminer_depth_cutoff = 0;
#define DEPTH_CUTOFF omp_get_num_threads()
char cutoff_test = 0;
#endif
#include "arvorequad.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
// #define DEBUG

void imprimePesoDeTodosOsQuadrantes(Quadrante *Q, int index) {
  taskminer_depth_cutoff++;
  if (Q == NULL)
    return;
  cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
  #pragma omp task untied default(shared) depend(inout:Q) if(cutoff_test)
  imprimePesoDeTodosOsQuadrantes(Q->NE, index * 4 + 1);
  cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
  #pragma omp task untied default(shared) depend(inout:Q[0][1]) if(cutoff_test)
  imprimePesoDeTodosOsQuadrantes(Q->NW, index * 4 + 2);
  cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
  #pragma omp task untied default(shared) depend(inout:Q[0][3]) if(cutoff_test)
  imprimePesoDeTodosOsQuadrantes(Q->SW, index * 4 + 3);
  cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
  #pragma omp task untied default(shared) depend(inout:Q[0][2]) if(cutoff_test)
  imprimePesoDeTodosOsQuadrantes(Q->SE, index * 4 + 4);
  #pragma omp taskwait
  Q->peso = getPesoTotalDoQuadrante(Q);
  printf("(%d, %d) %d\n", Q->centro.x, Q->centro.y, Q->peso);
taskminer_depth_cutoff--;
}

int main(int argc, char const *argv[]) {
  int maxLargura, x, y, peso, pesototal = 0;
  unsigned numPontos;
  scanf("%d", &maxLargura);
  scanf("%d", &numPontos);
  int debug = 0;
  if (numPontos < 20) {
    debug = 1;
  }
  ArvoreQuad *A = criaNovaArvoreQuad(maxLargura);
  while (numPontos > 0) {
    x = rand() % maxLargura;
    y = rand() % maxLargura;
    peso = rand() % 100;
    // scanf("%d %d %d", &x, &y, &peso);
    Ponto P;
    P.x = x;
    P.y = y;
    if (addNovaEstrelaNoPontoX(P, peso, A))
      pesototal += peso;
    numPontos--;
  }

  if (debug)
    cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
    #pragma omp parallel
    #pragma omp single
    #pragma omp task untied default(shared)
    imprimePesoDeTodosOsQuadrantes(A->raiz, 0);
#ifdef DEBUG
  if (pesototal != getPesoTotalDoQuadrante(A->raiz))
    printf("PESO INCORRETO\n");
#endif

  liberaMemoriaEmArvore(A);
  return 0;
}

Estrela *novaEstrela(Ponto P, int peso) {
  Estrela *star = (Estrela *)malloc(sizeof(Estrela));
  star->p = P;
  star->peso = peso;

  return star;
}

Quadrante *criaNovoQuadranteVazio(Ponto NE, Ponto NW, Ponto SE, Ponto SW) {
  Quadrante *Q = (Quadrante *)malloc(sizeof(Quadrante));
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
  int x_centro = (Q->limite_SE.x - Q->limite_SW.x) / 2 + Q->limite_SW.x;
  int y_centro = (Q->limite_NE.y - Q->limite_SE.y) / 2 + Q->limite_SE.y;
  Ponto centro;
  centro.x = x_centro;
  centro.y = y_centro;
  Q->centro = centro;

  return Q;
}

ArvoreQuad *criaNovaArvoreQuad(int larguraQuadranteRaiz) {
  ArvoreQuad *A = (ArvoreQuad *)malloc(sizeof(ArvoreQuad));
  Ponto centro, NE, NW, SE, SW;
  centro.x = larguraQuadranteRaiz / 2;
  centro.y = larguraQuadranteRaiz / 2;

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

int addNovaEstrelaNoPontoX(Ponto P, int peso, ArvoreQuad *A) {
  Estrela *nova = novaEstrela(P, peso);
  if (estrelaDentroDoQuadrante(nova, A->raiz)) {
    cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
    #pragma omp parallel
    #pragma omp single
    #pragma omp task untied default(shared)
    addEstrelaAoQuadranteRecursivamente(A->raiz, nova);
    return 1;
  }
  free(nova);
  return 0;
}

int estrelaDentroDoQuadrante(Estrela *star, Quadrante *Q) {
  Ponto P = star->p;
  //veririca se está dentro do quadrante
  if ((P.x >= Q->limite_SW.x && P.x < Q->limite_SE.x) &&
      (P.y >= Q->limite_SW.y && P.y < Q->limite_NW.y))
    return 1;

  return 0;
}

int calculaAreaDoQuadrante(Quadrante *Q) {
  int y = Q->limite_NE.y - Q->limite_SE.y;
  int x = Q->limite_SE.x - Q->limite_SW.x;

  return y * x;
}

void divideQuadrante(Quadrante *Q) {
  //TODO: não dividir se tiver área menor ou igual a 1.
  if (calculaAreaDoQuadrante(Q) <= 1) {
#ifdef DEBUG
    printf("Quadrante indivisível.\n");
#endif
    return;
  }

  //Define os limites dos novos quadrantes
  Ponto norte, sul, leste, oeste;
  norte.x = (Q->limite_NE.x - Q->limite_NW.x) / 2 + Q->limite_NW.x;
  norte.y = Q->limite_NE.y;

  sul.x = (Q->limite_SE.x - Q->limite_SW.x) / 2 + Q->limite_SW.x;
  sul.y = Q->limite_SE.y;

  leste.x = Q->limite_NE.x;
  leste.y = (Q->limite_NE.y - Q->limite_SE.y) / 2 + Q->limite_SE.y;

  oeste.x = Q->limite_NW.x;
  oeste.y = (Q->limite_NW.y - Q->limite_SW.y) / 2 + Q->limite_SW.y;

  Q->NE = criaNovoQuadranteVazio(Q->limite_NE, norte, leste, Q->centro);
  Q->NW = criaNovoQuadranteVazio(norte, Q->limite_NW, Q->centro, oeste);
  Q->SE = criaNovoQuadranteVazio(leste, Q->centro, Q->limite_SE, sul);
  Q->SW = criaNovoQuadranteVazio(Q->centro, oeste, sul, Q->limite_SW);
}

void addEstrelaAoQuadranteRecursivamente(Quadrante *Q, Estrela *star) {
  taskminer_depth_cutoff++;
  //ponto da Estrela p deve estar dentro do quadrante Q. Se não estiver, saia.
  //Faça uma função para verificar se o ponto está dentro do quadrante.
  if (!estrelaDentroDoQuadrante(star, Q))
    return;

#ifdef DEBUG
  printf("Tentando adicionar a Estrela no ponto: (%d, %d)\n", star->p.x, star->p.y);
#endif

  //Se o quadrante não pode mais ser dividido, retorne.
  if (calculaAreaDoQuadrante(Q) <= 1) {
    if (Q->star == NULL && Q->NE == NULL) {
      //Se está vazio e não tem filhos, adiciona a starícula e saia.
      Q->star = star;
      return;
    } else {
      Q->star->peso += star->peso;
    }
    return;
  }

  //Verifica se Q está vazio e não tem filhos
  if (Q->star == NULL && Q->NE == NULL) {
    //Se está vazio e não tem filhos, adiciona a starícula e saia.
    Q->star = star;
    return;
  } else {
    //Verifica se Q é dividido em subquadrantes
    if (Q->NE != NULL) //Qualquer filho não-nulo significa que Q é dividido
    {
      //tenta adicionar a starícula em todos os filhos
      cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
      #pragma omp task untied default(shared) depend(inout:star,Q) if(cutoff_test)
      addEstrelaAoQuadranteRecursivamente(Q->NE, star);
      cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
      #pragma omp task untied default(shared) depend(inout:star,Q[0][1]) if(cutoff_test)
      addEstrelaAoQuadranteRecursivamente(Q->NW, star);
      cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
      #pragma omp task untied default(shared) depend(inout:star,Q[0][2]) if(cutoff_test)
      addEstrelaAoQuadranteRecursivamente(Q->SE, star);
      cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
      #pragma omp task untied default(shared) depend(inout:star,Q[0][3]) if(cutoff_test)
      addEstrelaAoQuadranteRecursivamente(Q->SW, star);
    taskminer_depth_cutoff--;
    } else //se não for dividido, divide Q em subquadrantes
    {
      Estrela *aux = Q->star;

      //esvazia o quadrante atual
      Q->star = NULL;

      //Divide Q em subquadrantes
      divideQuadrante(Q);

      //Passa a starícula em Q para um quadrante abaixo
      cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
      #pragma omp task untied default(shared) depend(inout:aux,Q) if(cutoff_test)
      addEstrelaAoQuadranteRecursivamente(Q->NE, aux);
      cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
      #pragma omp task untied default(shared) depend(inout:Q[0][1],aux) if(cutoff_test)
      addEstrelaAoQuadranteRecursivamente(Q->NW, aux);
      cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
      #pragma omp task untied default(shared) depend(inout:aux,Q[0][2]) if(cutoff_test)
      addEstrelaAoQuadranteRecursivamente(Q->SE, aux);
      cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
      #pragma omp task untied default(shared) depend(inout:aux,Q[0][3]) if(cutoff_test)
      addEstrelaAoQuadranteRecursivamente(Q->SW, aux);

      //tenta adicionar a starícula em todos os filhos
      cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
      #pragma omp task untied default(shared) depend(inout:star,Q) if(cutoff_test)
      addEstrelaAoQuadranteRecursivamente(Q->NE, star);
      cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
      #pragma omp task untied default(shared) depend(inout:star,Q[0][1]) if(cutoff_test)
      addEstrelaAoQuadranteRecursivamente(Q->NW, star);
      cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
      #pragma omp task untied default(shared) depend(inout:star,Q[0][2]) if(cutoff_test)
      addEstrelaAoQuadranteRecursivamente(Q->SE, star);
      cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
      #pragma omp task untied default(shared) depend(inout:star,Q[0][3]) if(cutoff_test)
      addEstrelaAoQuadranteRecursivamente(Q->SW, star);
    #pragma omp taskwait
    }
  }
}

void liberaMemoriaEmArvore(ArvoreQuad *A) {
  cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
  #pragma omp parallel
  #pragma omp single
  #pragma omp task untied default(shared)
  liberaMemoriaEmQuadrantesRecursivamente(A->raiz);
  free(A);
}

void liberaMemoriaEmQuadrantesRecursivamente(Quadrante *Q) {
  taskminer_depth_cutoff++;
  if (Q == NULL)
    return;
  cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
  #pragma omp task untied default(shared) depend(in:Q) if(cutoff_test)
  liberaMemoriaEmQuadrantesRecursivamente(Q->NE);
  cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
  #pragma omp task untied default(shared) depend(in:Q[0][1]) if(cutoff_test)
  liberaMemoriaEmQuadrantesRecursivamente(Q->NW);
  cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
  #pragma omp task untied default(shared) depend(in:Q[0][2]) if(cutoff_test)
  liberaMemoriaEmQuadrantesRecursivamente(Q->SE);
  cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
  #pragma omp task untied default(shared) depend(in:Q[0][3]) if(cutoff_test)
  liberaMemoriaEmQuadrantesRecursivamente(Q->SW);
  #pragma omp taskwait
  if (Q->star != NULL) {
    free(Q->star);
  }
  if (Q != NULL)
    free(Q);

  taskminer_depth_cutoff--;
  return;
}

Quadrante *achaQuadranteParaEstrela(Ponto Estrela, ArvoreQuad *A) {
  return NULL;
}

int getPesoTotalDoQuadrante(Quadrante *Q) {
  taskminer_depth_cutoff++;
  if (Q == NULL)
    return 0;

  int peso;
  if (Q->star == NULL)
    peso = 0;
  else
    peso = Q->star->peso;

  cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
  taskminer_depth_cutoff--;
  #pragma omp task untied default(shared) depend(in:Q) if(cutoff_test)
  return getPesoTotalDoQuadrante(Q->NE) +
         cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
         #pragma omp task untied default(shared) depend(in:Q[0][2]) if(cutoff_test)
         getPesoTotalDoQuadrante(Q->SE) +
         cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
         #pragma omp task untied default(shared) depend(in:Q[0][1]) if(cutoff_test)
         getPesoTotalDoQuadrante(Q->NW) +
         cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
         #pragma omp task untied default(shared) depend(in:Q[0][3]) if(cutoff_test)
         getPesoTotalDoQuadrante(Q->SW) +
         #pragma omp taskwait
         peso;
}

