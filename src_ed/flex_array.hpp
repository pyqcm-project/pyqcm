#ifndef flex_array_h
#define flex_array_h

#include <iostream>
#include <cstdio>
#include <cstring>

#include "assert.h"

#define MIN_SIZE 16

//! arrays of undetermined, or growable length
/**
 This struct allows one to grow arrays on demand, by stringing together a linked list
 of short arrays of fixed length. Each instance of this struct is a segment of length segment_size,
 and contains pointers to the next segment and the current segment, i.e. the segment where
 new elements should be added.
 */
template<typename T>
struct flex_array
{

	size_t segment_size; //!< number of elements in each segment
	size_t pos;  //!<  current position within segment
	T *x; //!< pointer to segment data
	flex_array<T> *next; //!< pointer to next segment
	flex_array<T> *current;  //!< pointer to current segment
	
	//! Initializes the growing array with a growth size \a the_segment_size
	flex_array(size_t _segment_size): segment_size(_segment_size), pos(0L), x(nullptr), next(nullptr), current(nullptr)
	{
		if(segment_size < MIN_SIZE) segment_size = MIN_SIZE;
		x = (T *)malloc(segment_size*sizeof(T)); QCM_ASSERT(x);
		current = this;
	}
	
	
	
	//! Destructor
	~flex_array(){
		if(next) delete next;
		free(x);
		memset(this,0,sizeof(*this));
	}
	
	
	
	//! Adds an element elem to the current segment.
	void Put(T elem){
		if(current->pos == segment_size){
			current->next = new flex_array<T>(segment_size); QCM_ASSERT(current->next);
			current = current->next;
		}
		current->x[current->pos++] = elem;
	}
	
	
	
	/** Shrink the last segment to fit.
	 This method shrinks the last segment of the linked list to fit just the number of
	 elements in it. Used for memory efficiency in the case of short arrays that were
	 built with a larger than necessary grow size
	 */
	void streamline(){
		if(current->pos == segment_size) return;
		T *array;
		array = current->x;
		current->x = (T *)malloc(current->pos*sizeof(T)); QCM_ASSERT(current->x);
		memcpy(current->x,array,current->pos*sizeof(T));
		free(array);
	}
	
	
	
	//! Returns the number of elements in the array
	size_t size(){ // returns the number of elements in the array
		size_t n=pos;
		if(next) n += next->size();
		return n;
	}
	
	
	
};

#endif
