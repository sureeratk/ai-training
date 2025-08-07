// This example shows you how to train the word2vec model with your own content
// and leverage the model's nearest neighbor support. It also shows you how to
// use the cosine similarity algorithm to test similarity.
//
// # Running the example:
//
//	$ make example3
//
// # This requires running the following command:
//
//  $ make download-data // This will download and uncompress the data file.
//
// # Data File:
//
//  http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz
//
// # WARNING
//
//	This example uses a C++ based dynamic library that implements the Google
// 	word2vec model service. That dynamic library was pre-built by me. That
// 	library can be found under `foundation/word2vec/libw2v/lib/libw2v.dylib`.
//
// 	If you don't want to use that dynamic library, the instructions to build your
// 	own version exists here: https://github.com/fogfish/word2vec
//
// 	brew install cmake
// 	cd foundation/word2vec/libw2v
// 	cmake -DCMAKE_BUILD_TYPE=Release ../libw2v
// 	make

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"runtime"

	"github.com/ardanlabs/ai-training/foundation/stopwords"
	"github.com/ardanlabs/ai-training/foundation/vector"
	"github.com/ardanlabs/ai-training/foundation/word2vec"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	if err := cleanData(); err != nil {
		return fmt.Errorf("cleanData: %w", err)
	}

	if err := trainModel(); err != nil {
		return fmt.Errorf("trainModel: %w", err)
	}

	if err := testModel(); err != nil {
		return fmt.Errorf("trainModel: %w", err)
	}

	return nil
}

func cleanData() error {
	type document struct {
		ReviewText string
	}

	input, err := os.Open("zarf/data/example3.json")
	if err != nil {
		return fmt.Errorf("open file: %w", err)
	}
	defer input.Close()

	output, err := os.Create("zarf/data/example3.words")
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer output.Close()

	var counter int

	fmt.Print("\033[s")

	scanner := bufio.NewScanner(input)
	for scanner.Scan() {
		s := scanner.Text()

		var d document
		err := json.Unmarshal([]byte(s), &d)
		if err != nil {
			return fmt.Errorf("unmarshal: %w", err)
		}

		v := stopwords.Remove(d.ReviewText)

		output.WriteString(v)
		output.WriteString("\n")

		counter++

		fmt.Print("\033[u\033[K")
		fmt.Printf("Reading/Cleaning Data: %d", counter)
	}

	fmt.Print("\n")

	return nil
}

func trainModel() error {
	fmt.Println("Training Model ...")
	fmt.Print("\n")

	config := word2vec.Config{
		Corpus: word2vec.ConfigCorpus{
			InputFile: "zarf/data/example3.words",
			Tokenizer: " \n,.-!?:;/\"#$%&'()*+<=>@[]\\^_`{|}~\t\v\f\r",
			Sequencer: ".\n?!",
		},
		Vector: word2vec.ConfigWordVector{
			Vector:    300,
			Window:    5,
			Threshold: 1e-3,
			Frequency: 5,
		},
		Learning: word2vec.ConfigLearning{
			Epoch: 10,
			Rate:  0.05,
		},
		UseSkipGram:            false,
		UseCBOW:                true,
		UseNegativeSampling:    true,
		UseHierarchicalSoftMax: false,
		SizeNegativeSampling:   5,
		Threads:                runtime.GOMAXPROCS(0),
		Verbose:                true,
		Output:                 "zarf/data/example3.model",
	}

	if err := word2vec.Train(config); err != nil {
		return fmt.Errorf("train: %w", err)
	}

	fmt.Print("\n")

	return nil
}

func testModel() error {
	fmt.Println("Testing Model ...")
	fmt.Print("\n")

	w2v, err := word2vec.Load("zarf/data/example3.model", 300)
	if err != nil {
		return err
	}

	seq := make([]word2vec.Nearest, 10)
	w2v.Lookup("bad", seq)

	fmt.Println("Top 10 words similar to \"bad\"")
	fmt.Println(seq)
	fmt.Print("\n")

	// -------------------------------------------------------------------------

	words := []string{"terrible", "horrible", "price", "battery", "great", "nice"}

	for i := 0; i < len(words); i = i + 2 {
		var word1 [300]float32
		if err := w2v.VectorOf(words[i], word1[:]); err != nil {
			return err
		}

		var word2 [300]float32
		if err := w2v.VectorOf(words[i+1], word2[:]); err != nil {
			return err
		}

		v := vector.CosineSimilarity(word1[:], word2[:])

		fmt.Printf("The cosine similarity between the word %q and %q: %.3f%%\n", words[i], words[i+1], v*100)
	}

	return nil
}
