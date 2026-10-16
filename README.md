# Qwen3.8-Flash-Next → SpecForge DFlash2

Rozszerzenie notebooka GLM. Nowy, samowystarczalny notebook to
`notebooks/glm-qwen-flashnext-dflash2.ipynb`. Oryginał w Downloads pozostał bez zmian.

Pipeline obejmuje **prompty → odpowiedzi nauczyciela → dokładne tokeny i maski →
zbieranie cech lub obliczanie ich podczas treningu → trening i eksport DFlash2**.
Zbieranie danych i trening mają backend JAX, który pozwala wybrać wszystkie
lokalne urządzenia TPU. Trening wykonuje rzeczywisty model i funkcję straty
SpecForge przez Torchax; nie jest oddzielnym przybliżeniem DFlash2.

**Walidacja sprzętowa: lokalny CPU.** Mały Qwen4Exp zawiera rzeczywiste DeltaNet,
QSA, HC, PLE i MoE, lecz ma losowe wagi. Testy potwierdzają obliczenia i przepływ
danych. Nie potwierdzają jakości produkcyjnego draftera, szybkości na Kaggle ani
zmieszczenia pełnego checkpointu w konkretnym przydziale TPU/dysku.

## Szybki test na tym komputerze

Uruchamiaj z katalogu projektu:

```bash
bash scripts/bootstrap_cpu.sh
bash scripts/cpu_smoke.sh
.venv/bin/python -m pytest -q
```

Smoke test tworzy lokalny model i tokenizer, uruchamia dwóch workerów CPU,
generuje odpowiedzi z reasoning prefixem, zapisuje maski, pakuje wyniki ZIP,
scala korpus treningowy i walidacyjny, wykonuje trzy kroki treningu SpecForge
przez JAX i zapisuje `runs/cpu-smoke-*/job/training/draft`.
Nie pobiera pełnych wag Qwena i nie uruchamia GPU/TPU.

Ręczne wykonanie tego samego przykładu:

```bash
.venv/bin/qwen-dflash tiny-model --out runs/cpu-demo/target
.venv/bin/qwen-dflash run --config configs/job.cpu.json --stage worker --worker 0
.venv/bin/qwen-dflash run --config configs/job.cpu.json --stage worker --worker 1
.venv/bin/qwen-dflash run --config configs/job.cpu.json --stage train
```

Dwie komendy workerów można uruchomić jednocześnie. `--stage all` jest skrótem
dla pojedynczego workera, scalenia i treningu; przy wielu workerach pozostałe
artefakty muszą już być dostępne. Ponowne uruchomienie kontynuuje pracę.

## Dane: prompty i odpowiedzi tego nauczyciela

Obsługiwany jest lokalny JSONL z `prompt`, `input`, `text`, `messages` albo
ShareGPT `conversations`. Przykład:

```json
{"id":"p1","messages":[{"role":"user","content":"Napisz implementację sortowania przez scalanie."}]}
```

Można pobrać prompty strumieniowo z Hugging Face:

```bash
.venv/bin/qwen-dflash fetch-prompts --dataset nvidia/OpenCodeInstruct \
  --samples 5000 --seed 42 --out runs/prompts.jsonl
```

Pole `input` OpenCodeInstruct jest obsługiwane. Można podać `--revision`,
`--split` i `--dataset-config`. Zapis źródła zawiera rozwiązaną, niezmienną rewizję
repozytorium, liczbę próbek i SHA256. Użyj **tego samego przygotowanego JSONL** na
wszystkich instancjach: nie losuj osobno innego zbioru na każdym koncie.
Nie ma datasetu, który bez pomiarów gwarantuje najlepszego draftera Flash-Next.

Fetcher usuwa duplikaty promptów i końcową odpowiedź asystenta z cudzej rozmowy.
Poprzednie tury pozostają kontekstem. Nauczyciel generuje nową odpowiedź z własnym
chat template. `chat_template_kwargs.enable_thinking` ustawia tryb, jeżeli dany
tokenizer go obsługuje. Zapis zachowuje surowe tokeny i znaczniki reasoning;
maska ma 0 na prefiksie oraz 1 na nowo wygenerowanej odpowiedzi.
Nie dekodujemy odpowiedzi i nie tokenizujemy jej ponownie do treningu.

`temperature`, `top_p`, `top_k` i limit generowania są konfigurowalne.
Seed jest wyliczany osobno dla próbki, więc zmiana liczby rdzeni lub wznowienie
nie zmienia losowania danej odpowiedzi. Zbyt długi prompt zostaje jawnie wpisany
do rejestru pominięć. Dane nie są po cichu ucinane. Kontrola sum obejmuje zarówno
cechy, jak i rozmowy, także po przerwaniu pomiędzy ich zapisami.

## Wiele instancji Kaggle

Skopiuj `configs/job.kaggle-tpu.json`, dostosuj ścieżki do **kompletnego**
checkpointu oraz wspólnego `prompts.jsonl`. Konfiguracja pokazuje tryb TPU;
na tym komputerze używaj `configs/job.cpu.json`. Każda instancja ma identyczne
wejścia i konfigurację, ale inny numer workera:

```bash
qwen-dflash run --config job.json --stage worker --worker 0
# Druga instancja: --worker 1; kolejne: 2, 3, ...
```

Każdy worker sam przygotowuje identyczny plan i przetwarza swoją partycję.
Konta nie dzielą między sobą warstw modelu: każde potrzebuje wszystkich jego wag.
`workers` oznacza liczbę instancji, a `devices: "all"` liczbę lokalnych urządzeń
JAX użytych wewnątrz jednej instancji. Gdy próbek jest mniej niż rdzeni, część
rdzeni nie ma pracy; nie da się uniknąć tego samą flagą.

Worker zapisuje odpowiedzi i tokeny, opcjonalnie cechy, manifesty, sumy kontrolne
oraz `worker-N.zip`. Worker 0 zachowuje też embedding i LM head nauczyciela.
Przenieś ZIP-y wszystkich workerów na maszynę lub instancję treningową:

```bash
qwen-dflash run --config job.json --stage train \
  --artifacts /path/worker-0.zip /path/worker-1.zip /path/worker-2.zip /path/worker-3.zip
```

Etap `train` najpierw scala brakujące korpusy, następnie trenuje i eksportuje jeden
drafter. Dostępny jest też osobny `--stage merge`. Obsługiwane są ZIP-y i katalogi.
Przed scaleniem sprawdzane są wszystkie próbki, komplet workerów, wagi, tokenizer,
chat template, generowanie, warstwy i typ obliczeń. Braki oraz mieszanie różnych
nauczycieli powodują błąd. Scalony korpus jest niezależną kopią artefaktów.

Nie powstają niezależnie uczone „shardy draftera”, które można po prostu skleić.
Równoległe konta przyspieszają przygotowanie danych; finalny trening korzysta
ze wspólnego korpusu. Narzędzie nie loguje się na konta, nie publikuje datasetów
ani nie wymienia poświadczeń. Transport artefaktów wykonujesz sam.

## Limit czasu, zapis i wznowienie sesji

W notebooku ustaw w pierwszej komórce:

```python
MAX_RUN_MINUTES = 180
SAVE_MARGIN_MINUTES = 10
CHECKPOINT_EVERY_MINUTES = 5
RESUME_FROM = None
ARCHIVE_ON_PAUSE = True
```

To przykład własnego budżetu, bez odczytywania limitów konta Kaggle. Przy tych
ustawieniach pipeline przestaje rozpoczynać kolejne jednostki pracy po około
170 minutach, pozostawiając 10 minut na zapis. Notebook odejmuje czas od wykonania
komórki ustawień, również instalację. CLI liczy czas od rozpoczęcia komendy.
Ponowne wykonanie komórki ustawień rozpoczyna nowy licznik.

Odpowiednik w `job.json`:

```json
"runtime": {
  "max_run_seconds": 10800,
  "save_margin_seconds": 600,
  "checkpoint_interval_seconds": 300,
  "archive_on_pause": true,
  "resume_from": null
}
```

`max_run_seconds: null` wyłącza planowane zatrzymanie, pozostawiając okresowe
checkpointy. Parametry czasu można zmieniać między sesjami bez zmiany kontraktu
modelu/danych. Do zakończenia sesji stosowane są także obsługiwane sygnały
SIGINT/SIGTERM; obsługa następuje na bezpiecznej granicy pracy.

| Etap | Co zostaje zachowane |
|---|---|
| Przygotowanie promptów | Wybrane i odfiltrowane prompty, offset lokalnego pliku lub kursor strumienia HF, przypięta rewizja |
| Generowanie | Tokeny niedokończonej odpowiedzi i stan RNG; cache modelu jest odtwarzany z zachowanego prefiksu |
| Zbieranie cech | Atomowe pliki ukończonych przykładów i ich kwity; rozpoczęty forward trzeba dokończyć lub powtórzyć |
| Rozpakowywanie i scalanie | Ukończone pliki, sumy kontrolne oraz prefiks niedokończonej kopii |
| Trening | Ukończony krok, parametry, optymalizator/momenty, RNG i historia; rozpoczętego kroku nie zapisujemy połowicznie |
| Walidacja | Kursor następnego przykładu i dotychczasowe wyniki; nie powtarza ukończonej części walidacji |
| Konwersja wag | Ukończone fragmenty i kwity konwersji; ponownie uruchom tę samą komendę na tym samym katalogu |

Przy zatrzymaniu `run` zapisuje `session.json` ze statusem `paused`. Pozostawia
cały katalog zadania i domyślnie tworzy obok niego szybki ZIP bez kompresji:

```text
/kaggle/working/qwen-job/
/kaggle/working/qwen-job.resume-worker-0.zip
# albo: qwen-job.resume-train-0.zip / qwen-job.resume-merge-0.zip
```

W ZIP-ie są również plany, częściowe odpowiedzi, niedokończone scalenie i pliki
potrzebne do wznowienia. Nie zawiera pełnych wejściowych wag nauczyciela ani
środowiska Pythona. Pełne ZIP-y workerów `worker-N.zip` służą do scalania przez
`ARTIFACTS`; ZIP-y z `.resume-` służą do odtworzenia sesji przez `RESUME_FROM`.

**Przed zakończeniem sesji zachowaj ZIP lub cały katalog jako artefakt Kaggle.**
Sam zapis w `/kaggle/working` nie przenosi wyników do nowej sesji. Jeżeli ZIP
byłby zbyt duży albo zabrakłoby miejsca na dodatkową kopię, ustaw
`ARCHIVE_ON_PAUSE = False` i przenieś cały katalog, razem z plikami ukrytymi.
Nie wystarczy sam katalog `draft`: do kontynuowania potrzebny jest checkpoint
optymalizatora i stan pozostałych etapów.

W następnej sesji zamontuj zapisany wynik oraz oryginalny model i dane. Ustaw:

```python
MODE = "qwen_worker"  # ten sam etap co poprzednio, albo qwen_train
WORKER_ID = 0         # ten sam numer przy kontynuowaniu workera
RESUME_FROM = "/kaggle/input/my-progress/qwen-job.resume-worker-0.zip"
```

`config.out` powinien nadal wskazywać zapisywalny katalog, np.
`/kaggle/working/qwen-job`. Program odtworzy do niego pliki z wejściowego ZIP-a
lub katalogu. Zawartość modelu i danych musi być identyczna; nazwy ich mountów
mogą się zmienić. Przy niedokończonym scalaniu ponownie dołącz także źródłowe
ZIP-y workerów i ustaw `ARTIFACTS`. Po ukończeniu scalania trening korzysta już
z własnej kopii korpusu.

CLI obsługuje te same opcje:

```bash
qwen-dflash run --config job.json --stage worker --worker 0 \
  --max-run-seconds 10800 --save-margin-seconds 600 \
  --checkpoint-interval-seconds 300

qwen-dflash run --config job.json --stage worker --worker 0 \
  --resume-from /kaggle/input/my-progress/qwen-job.resume-worker-0.zip
```

W tym samym, zachowanym katalogu wystarczy ponownie uruchomić pierwotną komendę.
Flagi czasu działają też w `capture`, `merge`, `train`, `fetch-prompts` i
`prepare-weights`. Automatyczny ZIP całego zadania tworzy launcher `run`.

Limit jest kooperacyjny: nie przerywa w środku kompilacji/forwardu JAX, kroku
optymalizatora, hashowania dużego pliku ani końcowego zapisu/ZIP-a. Rezerwa musi
pomieścić najdłuższą taką operację i zapis. Nagłe odcięcie zasilania lub SIGKILL
pozwala odzyskać ostatni poprawny checkpoint, a nie pracę wykonaną od jego zapisu.
Odtworzenie strumienia HF może ponownie odczytać wcześniejszą część danych;
nie generuje ponownie już zapisanych odpowiedzi nauczyciela.

## Miejsce na cechy i wagi

`feature_mode: "tokens"` zapisuje małe pliki zawierające tylko `input_ids` i
`loss_mask`. Trener potrzebuje wtedy nauczyciela (`online_target`, ustawiany
automatycznie przez `run`) i przechwytuje cechy na bieżąco. Oszczędza to bardzo
dużo dysku kosztem ponownego liczenia nauczyciela w kolejnych epokach.

`feature_mode: "hidden_states"` zapisuje gotowe cechy SpecForge. Przy BF16 mają
one format BF16. Trening potrzebuje wtedy korpusu i zamrożonej głowy/embeddingu,
a pełny model nauczyciela nie musi być obecny. Duży korpus cech nadal może wymagać
wielu terabajtów: samych 5 warstw × 2560 × 4096 tokenów × 2 bajty to około
100 MiB na przykład, przed dodatkowym końcowym stanem.

Aby zmniejszyć magazyn pełnych wag, dostępna jest strumieniowa konwersja CPU:

```bash
qwen-dflash prepare-weights --model /path/full-bf16-model \
  --out /path/qwen-int8 --bits 8 --chunk-mib 32
# Bardziej stratny wariant: --bits 4
```

Własny format `qwen-stream-quant-v1` kwantyzuje macierze ekspertów, PLE i projekcji
w grupach po 64 wartości. Normy, metadane całkowite, embedding tokenów i LM head
zachowują oryginalną precyzję. Konwerter czyta fragmenty zamiast całego modelu,
zapisuje kwity postępu i po ponownym wywołaniu kontynuuje konwersję. Wielkość
pojedynczego wiersza/banku eksperta może przekroczyć zadany rozmiar fragmentu.
Odczyt otwiera ograniczoną liczbę plików i dekwantyzuje tylko potrzebne fragmenty.

Taki checkpoint działa przez **backend JAX tego projektu**, również jako
nauczyciel w pipeline. Nie jest GGUF ani formatem bezpośrednio obsługiwanym przez
`AutoModel.from_pretrained`. 4/8 bitów zmienia nauczyciela i może obniżyć jakość.
Nie mieszaj jego cech z BF16. Docelowe wdrożenie powinno używać tej samej wersji
nauczyciela, a jakość kwantyzacji trzeba ocenić na reprezentatywnych zadaniach.
Konwersja wymaga dostępu do źródłowych wag; nie wykonano jej dla pełnego modelu.

Wagi mogą być zamontowane z kilku datasetów: przygotuj katalog z indeksem
`model.safetensors.index.json` i linkami do odpowiednich shardów. Linki do danych
wejściowych są obsługiwane. Strumieniowanie i kwantyzacja zmniejszają wymagania,
ale nie gwarantują konkretnego limitu miejsca przyznanego przez Kaggle.

## Generowanie i długi kontekst

Natywny JAX obsługuje cache KV dla QSA, stan rekurencyjny DeltaNet, historię
konwolucji i historię n-gramów PLE. Kolejny token nie wymaga ponownego przeliczenia
całego prefiksu. Testy porównują kolejne kroki z pełnym przebiegiem Transformers,
także przy EOS wewnątrz sekwencji i na granicach mikrobloków QSA.

QSA wybiera i pobiera tylko potrzebne klucze/wartości, dzieląc zapytania na kafle.
Nie buduje pełnej maski ani macierzy attention `[S,S]`. Usunięto sztywny limit
2048: obowiązuje limit konfiguracji modelu lub jawny mniejszy `max_tokens` w API.
Test CPU obejmuje kontekst 2050 tokenów. To nie jest benchmark kontekstu 262k;
cache i wyszukiwanie bloków nadal kosztują pamięć oraz czas.

Zbieranie odpowiedzi/cech rozdziela różne próbki między wszystkie rdzenie.
Przy generowaniu pojedynczej sekwencji `generate --backend jax --devices all`
rozdziela aktywnych ekspertów MoE między rdzenie. Koordynator wykonuje attention,
DeltaNet, HC, PLE, shared expert i LM head. Aktywacje i prawdopodobieństwa
routingu są replikowane między urządzeniami JAX, a częściowe wyniki ekspertów
łączone przez `lax.psum`/all-reduce. Host odczytuje identyfikatory wybranych
ekspertów, aby wczytać odpowiednie wagi z dysku; nie odbiera ich aktywacji ani
częściowych wyników. Odczyt potrzebnych wierszy PLE także pozostaje na hoście.

## Optymalizacje TPU

Przykładowe konfiguracje włączają następujące opcje:

```json
"engine_options": {
  "qsa_backend": "jax",
  "query_chunk_size": 128,
  "kv_cache_block_size": 256,
  "expert_token_tile": 32,
  "weight_cache_mib": 1024,
  "lm_head_chunk_size": 16384,
  "compilation_cache_dir": "/kaggle/working/jax-compilation-cache"
}
```

- **QSA na urządzeniu:** wybór widocznych bloków, top-k, gather i attention są
  kompilowane jako kafelkowany kernel JAX/XLA. Nie ma kopiowania score'ów na CPU
  ani pętli Pythona po zapytaniach. `lax.map` ogranicza pamięć roboczą do kafla.
- **Fuzja obliczeń:** HC, RMSNorm, konwolucje i MLP eksperta korzystają z jąder
  JIT. Liczby tokenów przypisanych ekspertowi są dopełniane do wielokrotności
  `expert_token_tile`, co ogranicza liczbę wariantów kompilacji.
- **Cache wag:** budżet jest podany w MiB **na lokalne urządzenie i instancję
  silnika**. Małe, często używane wagi norm/konwolucji mają osobną chronioną
  część; większe macierze pozostają rezydentne, dopóki mieszczą się w budżecie.
  `0` wyłącza cache. Duży checkpoint nie musi zmieścić się w tym buforze.
  W trybie BF16 wagi są konwertowane przed przesłaniem na urządzenie.
- **Cache KV:** rośnie blokami; wewnątrz bloku kształty są stałe, a stare bufory
  są przekazywane do ponownego użycia przez XLA. Nie trzeba kompilować QSA
  osobno dla każdego nowego tokenu. RoPE nadal używa rzeczywistych pozycji.
- **LM head:** większe fragmenty projekcji i jeden transfer końcowych logitów
  zamiast transferu każdego małego fragmentu. Zwykły sampler pozostaje na CPU.
- **Trening:** aktualizacja Adam wykorzystuje donation buforów parametrów i
  momentów, a ich rozmieszczenie jest jawnie utrzymywane między krokami.
  W trybie tokenów nauczyciel przekazuje cechy bezpośrednio na urządzenia
  treningowe; nie odbywają drogi TPU → PyTorch CPU → TPU.
- **Cache kompilacji:** katalog można zachować jako artefakt i przywrócić w
  kolejnej zgodnej sesji. Rzeczywiste cache'owanie zależy również od progów JAX.

`qsa_backend: "auto"` wybiera JAX na TPU i referencję na CPU. Jawne `"jax"`
pozwala testować tę samą zoptymalizowaną ścieżkę na CPU. JAX rozstrzyga remisy
score'ów według niższego indeksu; PyTorch CPU top-k może wybrać inne równie dobre
bloki. Z tego powodu nowa polityka ma kontrakt `jax-streaming-v2` i zapis
`qsa_selection`. Nie mieszaj jej danych z `jax-streaming-v1`. Aby wznowić stary
korpus, ustaw `qsa_backend: "reference"`; nowy wariant uruchom w nowym katalogu.
Zwykły trening online sam odczytuje wymaganą politykę z kontraktu korpusu.

Testy obejmują referencyjne wyniki QSA, remisy, BF16, granice cache KV,
zgodność po zmianie liczby urządzeń oraz ośmiordzeniowy all-reduce. Kontrola
transferów zabrania kopiowania aktywacji ekspertów i cech nauczyciela na hosta.
Nadal nie ma własnych kerneli Pallas ani pomiaru pełnego nauczyciela na TPU.

Benchmark na podłączonym modelu, domyślnie CPU:

```bash
.venv/bin/python scripts/benchmark_tpu.py --model /path/to/shards \
  --device cpu --devices all --tokens 32 --new-tokens 8 --cache-mib 1024 \
  --out runs/benchmark-cpu.json
```

Na TPU zmień `--device tpu`. Skrypt porównuje referencję, QSA JAX bez cache wag
oraz QSA JAX z cache. Raport zawiera czasy po rozgrzewce i liczniki transferów
macierzy; nie jest testem jakości obu polityk top-k. Liczniki nie obejmują
odczytów pojedynczych wierszy embeddingu/PLE.

W pomiarze CPU na losowym modelu H=32, S=32, 8 generowanych tokenów uzyskano
około 21.7 tok/s dla referencji i 40.2 tok/s dla zoptymalizowanej ścieżki z cache.
To pojedynczy pomiar małego modelu; nie przewiduje wydajności pełnego Qwena/TPU.


## Trening na CPU i TPU

`training.backend: "jax"`, `device: "tpu"`, `devices: "all"` uruchamiają trening
SpecForge przez Torchax. Macierze parametrów, zamrożonego embeddingu/głowy i stany
Adam są dzielone między lokalne urządzenia przez JAX/GSPMD. Rozmieszczenie
pozostałych tensorów dobiera kompilator. `devices.json` zapisuje rzeczywiste kształty lokalnych shardów.
Ścieżka ta jest testowana na jednym i ośmiu logicznych urządzeniach CPU.
TPU wymaga zgodnego `jax[tpu]==0.8.3`; notebook ma osobną instalację runtime TPU.

Dostępne opcje: `epochs` albo `steps`, `batch_size`, `precision: "bfloat16"`,
`gradient_checkpointing`, `objective_chunk_blocks`, `warmup_steps`,
`checkpoint_every`, `anchors` i `sequence_length`. Mały `objective_chunk_blocks`
ogranicza tymczasowe logity dużego słownika. JAX używa stałej długości batcha,
aby nie kompilować programu osobno dla każdej długości. Krótsze próbki są
uzupełniane zerową maską; dłuższe powodują błąd zamiast obcięcia.

Domyślnie `run` wykonuje jedną epokę. To punkt startowy, nie gwarancja gotowego
produktu. Pipeline odkłada osobne prompty walidacyjne, wyklucza nakładanie próbek
i zapisuje `validation_loss`. Właściwą liczbę epok i anchorów wybieraj na podstawie
walidacji, a później akceptacji tokenów i przyspieszenia w docelowym serwerze.
Przykładowe 16 anchorów w konfiguracji TPU ogranicza koszty; nie jest deklaracją
jakości równoważnej pełnej recepturze upstream z 512 anchorami.

`steps` jest łączną docelową liczbą kroków. `run` sam wznawia `checkpoint.pt`.
Można zmienić liczbę lokalnych urządzeń JAX między wznowieniami. Zmiana danych,
architektury, precyzji albo istotnych opcji uczenia wymaga nowego katalogu.
Zakończony checkpoint można ponownie wyeksportować po utracie katalogu `draft`.

Backend PyTorch pozostaje dostępny na CPU, a launcher ma także opcje CUDA i
optymalizatora z kopiami FP32 na CPU (`optimizer_offload`). Obsługuje batchowanie,
BF16, checkpointing i DDP. Testowano CPU, także BF16/offload; CUDA nie uruchamiano.
Dla CPU DDP można użyć `torchrun --nproc-per-node=2`; między odrębnymi kontami
wymagałoby to działającej sieci synchronizacyjnej. Pipeline Kaggle jej nie zakłada.

## Adapter cech i eksport

Indeksy przechwytywanych warstw są zerowane. Surowy wynik warstwy Qwena ma
`hc_count × hidden_size`; adapter stosuje nauczony końcowy mixer HC nauczyciela
do każdej wybranej warstwy i łączy wyniki. Polityka
`qwen4_exp.post_decoder.final_hc_mixer.v1` jest zapisana w kontrakcie. To jawny
wybór projektowy wymagający oceny jakości na pełnym nauczycielu.

Drafter to rzeczywisty `DFlash2DraftModel`, łącznie z konwolucjami i selektorem
kandydatów. Używana jest upstreamowa funkcja straty `OnlineDFlashModel`.
Eksport zawiera config, safetensors i `target_contract.json`. Wdrożenie w
SGLang/vLLM nadal potrzebuje adaptera tej samej polityki HC i obsługi Flash-Next
w konkretnym serwerze. Nie rozszerzano teraz dekodowania spekulacyjnego;
wcześniejszy referencyjny wariant greedy pozostał dostępny.

## Źródła i odtwarzalność

- [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next);
  oryginalna konfiguracja: `configs/qwen3.8-flash-next.original.json`.
- [Transformers Qwen4Exp](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen4_exp),
  testowana wersja `5.19.0`.
- [SpecForge](https://github.com/sgl-project/SpecForge/tree/d80daf31aacfd2193283ae36f36864e4dde199bc),
  rewizja `d80daf31aacfd2193283ae36f36864e4dde199bc`.
- [Torchax](https://github.com/google/torchax/tree/d95b0a5bff13e2b945e6a0e7ba3b3937b838161b),
  rewizja `d95b0a5bff13e2b945e6a0e7ba3b3937b838161b`.
  Przypięta wersja źródłowa jest zgodna z użytym PyTorch 2.14.1.
- [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) jako
  opcjonalne źródło promptów, bez przejmowania jego odpowiedzi jako nowej odpowiedzi nauczyciela.
- [kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab), pochodzenie modułów GLM;
  oryginalne moduły i notebook zachowane w `legacy/`, licencje w `third_party/`.

Notebook osadza źródła projektu, SpecForge i Torchax. `scripts/build_notebook.py`
odbudowuje go ze źródeł. Oryginalne komórki GLM wykonują się wyłącznie po wybraniu
`glm53_legacy`. Podczas tej pracy nie uruchamiano ich serwera, tunelu ani keepalive.

Implementacja optymalizacji korzysta z publicznych mechanizmów JAX:
[top-k](https://docs.jax.dev/en/latest/_autosummary/jax.lax.top_k.html),
[shard_map i kolektywów](https://docs.jax.dev/en/latest/notebooks/shard_map.html).
