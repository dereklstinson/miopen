package miopen_test

import (
	"fmt"
	"testing"

	miopen "github.com/dereklstinson/migo"
)

func TestHandle(t *testing.T) {
	handle := miopen.CreateHandle()
	fmt.Println(handle)

}
