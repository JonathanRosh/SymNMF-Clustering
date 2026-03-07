CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors

symnmf: symnmf.c symnmf.h
	$(CC) $(CFLAGS) symnmf.c -o symnmf -lm