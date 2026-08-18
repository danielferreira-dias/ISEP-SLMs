Tarefa: desenhar e implementar uma experiência E4 de supervised knowledge distillation com temperatura adaptativa por amostra.

Contexto:
- E3 usa hard KD answer-blind: o Teacher gera targets textuais/estruturados Stage A/B.
- E4 será uma experiência separada de logit-level KD para classificação.
- O Teacher fica frozen/eval.
- Apenas o Student recebe gradients.
- Teacher e Student têm de usar exatamente a mesma taxonomia de classes.
- O ground truth nunca é enviado ao Teacher; é usado apenas na CE e na ponderação posterior da KD.
- Não alterar silenciosamente o E3.

Objetivo:
Treinar o Student com ground-truth CE e soft targets do Teacher, dando maior influência ao Teacher quando a sua previsão é fiável e reduzindo a influência quando está errado ou incerto.

Loss por amostra:

L_i = CE(y_i, z_S,i)
      + λ_i · T_i² ·
        KL(
          softmax(z_T,i / T_i)
          ||
          softmax(z_S,i / T_i)
        )

O fator T_i² deve ser mantido para compensar a redução dos gradientes causada pela temperatura.

Peso adaptativo da KD:

Baseline binário:

λ_i = λ_correct, se argmax(z_T,i) == y_i
λ_i = λ_wrong, caso contrário

Configuração inicial:
- λ_correct = 1.0
- λ_wrong = 0.1

Variante recomendada, contínua:

1. Calcular uma distribuição calibrada do Teacher com uma temperatura fixa de calibração T_cal.
2. Obter p_y = probabilidade atribuída pelo Teacher ao ground-truth.
3. Calcular correct_i = 1[argmax(z_T,i) == y_i].
4. Usar:

λ_i =
  λ_wrong
  + (λ_correct - λ_wrong)
    · correct_i
    · p_y

Isto dá peso elevado a Teachers corretos e confiantes, mantendo peso baixo quando o Teacher está errado.

Temperatura adaptativa — variante principal:

A temperatura deve depender da incerteza do Teacher, não diretamente do facto de estar certo ou errado.

1. Calcular:

q_i = softmax(z_T,i / T_cal)

2. Entropia normalizada:

u_i =
  -sum_c q_i,c log(q_i,c)
  / log(num_classes)

3. Temperatura:

T_i =
  clip(
    T_min + u_i · (T_max - T_min),
    T_min,
    T_max
  )

Interpretação:
- Teacher confiante, baixa entropia → temperatura baixa;
- Teacher ambíguo, alta entropia → temperatura alta;
- temperatura alta suaviza os soft targets e preserva alternativas plausíveis.

Configuração inicial sugerida:
- T_cal = 1.0
- T_min = 1.0
- T_max = 4.0
- temperatura global baseline = 2.0

Variante secundária baseada na escala dos logits:

s_i = std(z_T,i)
T_i = clip(T_0 · s_i / s_ref, T_min, T_max)

`s_ref` deve ser estimado num conjunto de calibração separado. Esta variante deve ser tratada como ablação, porque a escala bruta dos logits pode refletir má calibração e não dificuldade visual real.

Implementação:
- Teacher em `eval()`;
- Teacher forward dentro de `torch.no_grad()`;
- `teacher_logits = teacher_logits.detach().float()`;
- Student logits em float para estabilidade;
- CE calculada com logits Student sem temperatura;
- KD calculada com uma temperatura diferente por amostra;
- usar operações vetorizadas sobre o batch;
- não fazer loops por classe ou por amostra;
- se o Teacher for generativo, confirmar que há logits de classes verdadeiros; probabilidades de tokens dos nomes das doenças são apenas uma aproximação e devem ser identificadas como tal.

Pseudocódigo:

Teacher.eval()
freeze(Teacher)

for images, labels in dataloader:

    with no_grad():
        teacher_logits = Teacher(images)

    student_logits = Student(images)

    q_cal = softmax(teacher_logits / T_cal)
    entropy = normalized_entropy(q_cal)

    T_i = clamp(
        T_min + entropy * (T_max - T_min),
        T_min,
        T_max
    )

    teacher_correct = argmax(teacher_logits) == labels
    teacher_gold_probability = q_cal[range(batch), labels]

    lambda_i = lambda_wrong + (
        lambda_correct - lambda_wrong
    ) * teacher_correct * teacher_gold_probability

    p_teacher = softmax(teacher_logits / T_i[:, None])
    log_p_student = log_softmax(student_logits / T_i[:, None])

    kd_i = T_i² * sum(
        p_teacher * (log(p_teacher) - log_p_student),
        dim=classes
    )

    ce_i = cross_entropy(student_logits, labels)

    loss = mean(ce_i + lambda_i * kd_i)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

Registar por batch/epoch:
- CE média;
- KD média;
- loss total;
- accuracy do Teacher;
- probabilidade média do Teacher no label gold;
- λ médio;
- temperatura média, mínima e máxima;
- entropia média;
- accuracy do Student;
- resultados separados quando Teacher está correto/incorreto;
- Top-1, Top-3, macro-F1 e balanced accuracy;
- ECE/Brier/NLL para calibração;
- latência, VRAM e custo de inferência do Teacher.

Ablations obrigatórias:

1. CE only.
2. CE + KD com temperatura global fixa.
3. CE + KD com λ binário correcto/incorreto.
4. CE + KD com λ contínuo baseado em p_T(y).
5. CE + KD com λ contínuo + temperatura adaptativa por entropia.
6. Opcional: temperatura adaptativa baseada em escala dos logits.

Para atribuição causal:
- manter o mesmo Student inicial;
- manter o mesmo dataset, split, seed, LR, LoRA topology e número de passos;
- incluir um braço CE-only com o mesmo treino adicional;
- não escolher checkpoints usando o resultado do próprio benchmark final;
- calibrar T_cal e s_ref apenas num conjunto de calibração separado.

Interpretação:
- ganho simultâneo em accuracy e calibração → evidência favorável à KD adaptativa;
- ganho apenas em calibração/Top-3 → melhoria de incerteza sem ganho Top-1;
- degradação quando Teacher está errado → λ_wrong demasiado alto ou Teacher mal calibrado;
- KD sem ganho sobre CE → Teacher pode não acrescentar informação transferível ou Student pode não ter capacidade suficiente.

Nome sugerido:
E4 — Confidence-Aware Adaptive Logit Distillation.