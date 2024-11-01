#include "Common.h"
#include "Shader.h"

void glEnable2D(void)
{
	GLint iViewport[4];

	glGetIntegerv( GL_VIEWPORT, iViewport );

	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();

	glOrtho( iViewport[0], iViewport[0]+iViewport[2], iViewport[1]+iViewport[3], iViewport[1], -1, 1 );
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();    

	glPushAttrib( GL_DEPTH_BUFFER_BIT | GL_LIGHTING_BIT );
	glDisable( GL_DEPTH_TEST );
	glDisable( GL_LIGHTING );
}

void glDisable2D(void)
{
	glPopAttrib();
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();
}

void glEnableFixedFunction(const Transform &trans)
{
    const mat4 projT = transpose(trans.projection);
    const mat4 viewT = transpose(trans.view);

    float pm[16];
    float mvm[16];

    projT.data(pm);
    viewT.data(mvm);

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(pm);

    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(mvm);
}

void glDisableFixedFunction()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

float cosineInterpolation(float a, float b, float s)
{
   float s2;

   s2 = (float)( 1 - cos(s * math_pi) ) / 2;

   return float((a * (1 - s2) + b * s2));
}

float hermiteInterpolation(float y0, float y1, float y2, float y3, float mu, float tension, float bias)
{
    float m0,m1,mu2,mu3;
    float a0,a1,a2,a3;

    mu2 = mu * mu;
    mu3 = mu2 * mu;
    m0  = (y1-y0)*(1+bias)*(1-tension)/2;
    m0 += (y2-y1)*(1-bias)*(1-tension)/2;
    m1  = (y2-y1)*(1+bias)*(1-tension)/2;
    m1 += (y3-y2)*(1-bias)*(1-tension)/2;
    a0 =  2*mu3 - 3*mu2 + 1;
    a1 =    mu3 - 2*mu2 + mu;
    a2 =    mu3 -   mu2;
    a3 = -2*mu3 + 3*mu2;

    return(a0*y1+a1*m0+a2*m1+a3*y2);
}

void renderTexture(uint texture, uint posX, uint posY, float width, float height)
{   
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glColor4f(0.0, 0.0, 0.0, 1.0f);  
    
    glEnable(GL_TEXTURE_2D);     
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    //glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE); 

    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16.0f);


    glDisable(GL_DEPTH_TEST);
    
    
    glEnable2D();
    glPushMatrix();            
        glTranslatef((float)posX, (float)posY, 0.0f);
        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 1.0f);
            glVertex3f(0.0f, 0.0f, 0.0f);

            glTexCoord2f(1.0f, 1.0f);
            glVertex3f(width, 0.0f, 0.0f);
            
            glTexCoord2f(1.0f, 0.0f);
            glVertex3f(width, height, 0.0f);

            glTexCoord2f(0.0f, 0.0f);
            glVertex3f(0.0, height, 0.0f);
        glEnd();
    glPopMatrix();
    glDisable2D();

    glBindTexture(GL_TEXTURE_2D, 0);

    glDisable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);   
}

void renderQuad(float size, float r, float g, float b, float a)
{
    float w = size;
    float h = size;
    float d = 0.0;

    glColor4f(r, g, b, a);
    glBegin(GL_QUADS);
        glVertex3f(-w, -h, -d);
        glVertex3f( w, -h, -d);
        glVertex3f( w,  h, -d);
        glVertex3f(-w,  h, -d);
    glEnd();
}

void renderQuad(float width, float height, float r, float g, float b, float a)
{
    float w = width;
    float h = height;
    float d = 0.0;

    glColor4f(r, g, b, a);
    glBegin(GL_QUADS);
        glVertex3f(-w, -h, -d);
        glVertex3f( w, -h, -d);
        glVertex3f( w,  h, -d);
        glVertex3f(-w,  h, -d);
    glEnd();
}

void renderQuad(float posX, float posY, float width, float height)
{
    glPushMatrix();            
        glTranslatef(posX, posY, 0.0f);       
        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 1.0f);
            glVertex3f(0.0f, 0.0f, 0.0f);

            glTexCoord2f(1.0f, 1.0f);
            glVertex3f(width, 0.0f, 0.0f);
            
            glTexCoord2f(1.0f, 0.0f);
            glVertex3f(width, height, 0.0f);

            glTexCoord2f(0.0f, 0.0f);
            glVertex3f(0.0, height, 0.0f);
        glEnd();
    glPopMatrix();
}

void renderOrigin(float lineWidth)
{
    float size = 0.5;

    glLineWidth(lineWidth);
    glPushMatrix();
        glBegin(GL_LINES);
            glColor3f(1.0f, 0.0f, 0.0f);
            glVertex3f(0.0f, 0.0f, 0.0f);
            glVertex3f(size, 0.0f, 0.0f);

            glColor3f(0.0f, 1.0f, 0.0f);
            glVertex3f(0.0f, 0.0f, 0.0f);
            glVertex3f(0.0f, size, 0.0f);

            glColor3f(0.0f, 0.0f, 1.0f);
            glVertex3f(0.0f, 0.0f, 0.0f);
            glVertex3f(0.0f, 0.0f, size);
        glEnd();
    glPopMatrix(); 
    glLineWidth(1.0f);
}

void screenSizeQuad(float width, float height, float fov)
{
    //Screen Size Quad in 3D
    float phi = fov * 0.5f;
    float d = 1.0f;

    float h = 2.0f * float(tan( (double) phi/180.0 * 3.141592654) * d);
    float w = h * width/height;

    w *= 0.5f;
    h *= 0.5f;    

    glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-w, -h, -d);

        glTexCoord2f(1.0, 0.0);
        glVertex3f( w, -h, -d);
        
        glTexCoord2f(1.0, 1.0);
        glVertex3f( w,  h, -d);

        glTexCoord2f(0.0, 1.0);
        glVertex3f(-w,  h, -d);
    glEnd(); 
}

void renderString(const char *str, int x, int y, vec4 color, void *font)
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_TEXTURE_2D);

    glEnable2D();

        glColor4f(color.x, color.y, color.z, color.w);
        glRasterPos2i(x, y);

        while(*str)
        {
            glutBitmapCharacter(font, *str);
            ++str;
        }

    glDisable2D();

    glPopClientAttrib();
    glPopAttrib();
}

void renderString(const char *str, int x, int y, float r, float g, float b, float a, void *font)
{
    vec4 color = vec4(r, g, b, a);
    renderString(str, x, y, color, font);
}

void smoothBackground(vec4 top, vec4 bottom, float windowWidth, float windowHeight)
{
    glEnable2D();        
    glBegin(GL_QUADS);
        glColor4f(top.x, top.y, top.z, top.w);
        glVertex2f(windowWidth, 0.0);
        glVertex2f(0.0, 0.0);        

        glColor4f(bottom.x, bottom.y, bottom.z, bottom.w);
        glVertex2f(0.0, windowHeight);        
        glVertex2f(windowWidth, windowHeight);        
    glEnd(); 
    glDisable2D();
}

//void saveFrameBuffer(QGLWidget *widget)
//{
    //QImage img = widget->grabFrameBuffer(true);

    //QDate date = QDate::currentDate(); 
    //QTime time = QTime::currentTime();

    //int year = date.year();
    //int month = date.month();
    //int day = date.day();

    //int hour = time.hour();
    //int minute = time.minute();
    //int second = time.second();

    //QString sYear   = QString::number(year);
    //QString sMonth  = QString::number(month);
    //QString sDay    = QString::number(day);
    //QString sHour   = QString::number(hour);
    //QString sMinute = QString::number(minute);
    //QString sSecond = QString::number(second);
    //QString sNull   = QString::number(0);

    //QString fMonth  = month < 10 ? sNull + sMonth : sMonth;
    //QString fDay    = day < 10 ? sNull + sDay : sDay;
    //QString fHour   = hour < 10 ? sNull + sHour : sHour;
    //QString fMinute = minute < 10 ? sNull + sMinute : sMinute;
    //QString fSecond = second < 10 ? sNull + sSecond : sSecond;

    //QString fileName = sYear + fMonth + fDay + "_" + fHour + fMinute + fSecond + ".jpg";

    //img.save(fileName, "jpg", 100);
//}

//void saveFrameBuffer(QGLWidget *widget, int idx)
//{
    //QImage img = widget->grabFrameBuffer(true);

    //QDate date = QDate::currentDate(); 
    //QTime time = QTime::currentTime();

    //int year = date.year();
    //int month = date.month();
    //int day = date.day();

    //int hour = time.hour();
    //int minute = time.minute();
    //int second = time.second();

    //QString number = QString("%1").arg(idx, 5, 10, QChar('0'));

    //QString sYear   = QString::number(year);
    //QString sMonth  = QString::number(month);
    //QString sDay    = QString::number(day);
    //QString sHour   = QString::number(hour);
    //QString sMinute = QString::number(minute);
    //QString sSecond = QString::number(second);
    //QString sNull   = QString::number(0);

    //QString fMonth  = month < 10 ? sNull + sMonth : sMonth;
    //QString fDay    = day < 10 ? sNull + sDay : sDay;
    //QString fHour   = hour < 10 ? sNull + sHour : sHour;
    //QString fMinute = minute < 10 ? sNull + sMinute : sMinute;
    //QString fSecond = second < 10 ? sNull + sSecond : sSecond;

    ////QString fileName = "Output/" + sYear + fMonth + fDay + "_" + fHour + fMinute + fSecond + "_" + number + ".jpg";
    //QString fileName = "Output/" + number + ".jpg";

    //img.save(fileName, "jpg", 100);
//}

void getCameraFrame(const Transform &trans, vec3 &dir, vec3 &up, vec3 &right, vec3 &pos)
{
    Matrix4x4 view = trans.view;

    up = normalize(vec3(view.a21, view.a22, view.a23));
    right = normalize(vec3(view.a11, view.a12, view.a13));
    dir = normalize(cross(up, right));

    pos.x = -(view.a11 * view.a41 + view.a12 * view.a42 + view.a13 * view.a43);
    pos.y = -(view.a21 * view.a41 + view.a22 * view.a42 + view.a23 * view.a43);
    pos.z = -(view.a31 * view.a41 + view.a32 * view.a42 + view.a33 * view.a43);
}

vec3 getCamPosFromModelView(const Transform &trans)
{
	mat4 m = trans.view;
	vec3 c;

	mat4 inv = m.inverse();

	/*c.x = -(m.a11 * m.a43 + m.a21 * m.a24 + m.a31 * m.a34);
	c.y = -(m.a12 * m.a43 + m.a22 * m.a24 + m.a32 * m.a34);
	c.z = -(m.a13 * m.a43 + m.a23 * m.a24 + m.a33 * m.a34);*/

	c.x = inv.a14;
	c.y = inv.a24;
	c.z = inv.a34;

	return c;
}

vec3 getViewDirFromModelView(const Transform &trans)
{
	mat4 m = trans.view;
	vec3 c;

	c.x = -m.a31;
	c.y = -m.a32;
	c.z = -m.a33;

	return normalize(c);
}

vec3 getUpDirFromModelView(const Transform &trans)
{
	mat4 m = trans.view;
	vec3 c;

	c.x = m.a21;
	c.y = m.a22;
	c.z = m.a23;

	return normalize(c);
}

void colorMap(float x, float * out, float * cm)

{
	x = clamp(x*63, 0.f, 63.f - (float)1e-6);
	int idx = int(x);
	float r = fract(x);
	out[0] = cm[idx*3+0]*(1-r) + cm[(idx+1)*3+0]*r;
	out[1] = cm[idx*3+1]*(1-r) + cm[(idx+1)*3+1]*r;
	out[2] = cm[idx*3+2]*(1-r) + cm[(idx+1)*3+2]*r;
}

void colorMapBgr(float x, float * out, float * cm)
{
	x = clamp(x*63, 0.f, 63.f - (float)1e-6);
	int idx = int(x);
	float r = fract(x);
	out[2] = cm[idx*3+0]*(1-r) + cm[(idx+1)*3+0]*r;
	out[1] = cm[idx*3+1]*(1-r) + cm[(idx+1)*3+1]*r;
	out[0] = cm[idx*3+2]*(1-r) + cm[(idx+1)*3+2]*r;
}

void clearInputs()
{
    inputs::inst()->mouse_diff = vec2(0.0f, 0.0f);
    inputs::inst()->scroll_x = 0.0f;
    inputs::inst()->scroll_y = 0.0f;

    //inputs::inst()->gui_clicked = false;

    //inputs::inst()->w = false;
    //inputs::inst()->a = false;
    //inputs::inst()->s = false;
    //inputs::inst()->d = false;
}