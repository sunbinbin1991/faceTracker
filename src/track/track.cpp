#include "track.h"
track::track(/* args */)
{
	bool flagtemp = flag.load();
	printf("%d\n", flagtemp);
}

track::~track()
{
}  