#pragma once

#include <omp.h>

struct Timer {
	double last, total;

public:
	void tick() {
		last = omp_get_wtime();
	}

	void tock() {
		double now = omp_get_wtime();
		total = now - last;
	}

	void output(char* msg) {
		tm_printf(L"%hs: %3.5f s\n", msg, total);
	}
};


# define	TIMING_BEGIN(message) \
{tm_printf(L"%hs\n", message); Timer _c; _c.tick();

# define	TIMING_END(message) \
{_c.tock(); _c.output(message);}}

