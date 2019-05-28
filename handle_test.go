package miopen_test

import (
	"fmt"
	"testing"

	"github.com/dereklstinson/miopen"
)

func TestHandle(t *testing.T) {
	handle := miopen.CreateHandle(true)
	fmt.Println(handle)

}
