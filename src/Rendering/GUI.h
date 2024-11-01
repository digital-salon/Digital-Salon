#ifndef GRAPHICALUSERINTERFACE_H
#define GRAPHICALUSERINTERFACE_H

#include "Common.h"
#include "Texture.h"

class CameraManager;
class VertexBufferObject;
class Shader;
class GUIElement;
class Scene;

class GUIElement
{
public:
    GUIElement(){};
    virtual ~GUIElement(){};

   virtual void render() = 0;
   virtual bool onMouseClick(uint mx, uint my) = 0;
   virtual void onMouseMove(uint mx, uint my) = 0;
   virtual void onMouseRelease() = 0;

private:  
    
};

class ComboBox : public GUIElement
{
public:
    ComboBox(uint px, uint py, uint w, uint h, string text)
    : m_width(w), 
      m_height(h),
      m_posX(px),
      m_posY(py),
      m_unfold(false),
      m_mouseClick(false),
      m_text(text),
      m_variable(NULL),
      m_activeIdx(0),
      m_activeText("NULL"),
      rowHeight(20),
      m_hooverId(-1),
      m_color(1.0f, 1.0f, 1.0f, 1.0f)
    {    
    }

    ~ComboBox()
    {
    }

    void render()
    {
        glEnable2D();

        float x1 = float(m_posX);
        float x2 = float(m_posX + m_width);
        float y1 = float(m_posY);
        float y2 = float(m_posY + m_height);
    
        glColor4f(m_color.x, m_color.y, m_color.z,  m_color.w);
    
        glBegin(GL_LINES);
            glVertex2f(x1, y1);
            glVertex2f(x2, y1);

            glVertex2f(x1-1, y2);
            glVertex2f(x2, y2);
        
            glVertex2f(x1, y1);
            glVertex2f(x1, y2);

            glVertex2f(x2, y1);
            glVertex2f(x2, y2);
        glEnd();


        float qx1 = float(m_posX + m_width - 20);
        float qx2 = float(m_posX + m_width - 2);
        float qy1 = float(m_posY + 2);
        float qy2 = float(m_posY + m_height - 1);    
  
        glBegin(GL_QUADS);
            glVertex2f(qx1, qy1);
            glVertex2f(qx1, qy2);
            glVertex2f(qx2, qy2);
            glVertex2f(qx2, qy1);
        glEnd();

        //if(m_unfold)
        //{
        //    float sx1 = m_posX;
        //    float sx2 = m_posX + m_width;
        //    float sy1 = m_posY + m_height + 3;
        //    float sy2 = m_posY + m_height + 3 + rowHeight * m_items.size();

        //    glBegin(GL_LINES);
        //        glVertex2f(sx1, sy1);
        //        glVertex2f(sx2, sy1);

        //        glVertex2f(sx1-1, sy2);
        //        glVertex2f(sx2, sy2);
        //    
        //        glVertex2f(sx1, sy1);
        //        glVertex2f(sx1, sy2);

        //        glVertex2f(sx2, sy1);
        //        glVertex2f(sx2, sy2);
        //    glEnd();

        //    if(m_hooverId >= 0)
        //    {
        //        glColor4f(m_color.x, m_color.y, m_color.z,  m_color.w);
        //        float hx1 = m_posX+1;
        //        float hx2 = m_posX + m_width - 2;
        //        //float hy1 = m_posY + m_height + 2 + 13 + rowHeight * m_hooverId;
        //        //float hy2 = m_posY + m_height + 2 + 13 + rowHeight * m_hooverId + 10 ;    
        //        int y = m_posY + m_height + 5+ rowHeight * (m_hooverId);
        //  
        //        glBegin(GL_QUADS);
        //            glVertex2f(hx1, y);
        //            glVertex2f(hx1, y+rowHeight-3);
        //            glVertex2f(hx2, y+rowHeight-3);
        //            glVertex2f(hx2, y);
        //        glEnd();
        //    }

        //    int i = 0;
        //    //for(QMap<int, QString>::iterator iter = m_items.begin(); iter != m_items.end(); ++iter)
        //    //{
        //    //    QString text = iter.value(); 

        //    //    renderString(text.toStdString().c_str(), sx1 + 5, sy1 + rowHeight - 5 + (rowHeight)*i, m_color.x, m_color.y, m_color.z, m_color.w, GLUT_BITMAP_HELVETICA_12);
        //    //    i++;
        //    //}

        //}

        glDisable2D();

        string sliderText = m_activeText;
        renderString(m_text.c_str(), m_posX - 2, m_posY - 5, m_color.x, m_color.y, m_color.z, m_color.w, GLUT_BITMAP_HELVETICA_12);
        renderString(sliderText.c_str(), m_posX+5, m_posY+14, m_color.x, m_color.y, m_color.z, m_color.w, GLUT_BITMAP_HELVETICA_12);
    }

    void onMouseMove(uint mx, uint my)
    {
        m_hooverId = -1;
        if(m_unfold)
        {
            int i=0; 
            //for(QMap<int, QString>::iterator iter = m_items.begin(); iter != m_items.end(); ++iter)
            //{
            //    float sx1 = m_posX;
            //    float sx2 = m_posX + m_width;
      
            //    int y = m_posY + m_height + 5+ rowHeight * (i);

            //    if(mx > sx1 && mx < sx2)
            //    {
            //        if(my > uint(y) && my < uint(y+rowHeight))
            //        {    
            //            m_hooverId = i;
            //        }
            //    }

            //    i++;
            //}
        }
    }

    bool onMouseClick(uint mx, uint my)
    {
        float qx1 = float(m_posX + m_width - 20);
        float qx2 = float(m_posX + m_width - 2);
        float qy1 = float(m_posY + 2);
        float qy2 = float(m_posY + m_height - 1);  

        m_mouseClick = false;

        if(!m_unfold)
        {
            if(mx > qx1 && mx < qx2)
            {
                if(my > qy1 && my < qy2)
                {    
                    m_mouseClick = true;
                    m_unfold = true;
                }
            }
        }
        else
        {
            if(mx > qx1 && mx < qx2)
            {
                if(my > qy1 && my < qy2)
                {    
                    m_mouseClick = false;
                    m_unfold = false;
                }
            }
            else
            {
                //int i=0;
                //for(QMap<int, QString>::iterator iter = m_items.begin(); iter != m_items.end(); ++iter)
                //{
                //    float sx1 = m_posX;
                //    float sx2 = m_posX + m_width;
                //    int y = m_posY + rowHeight + (rowHeight) * i;
                //
                //    if(mx > sx1 && mx < sx2)
                //    {
                //        if(my > uint(y) && my < uint(y+rowHeight))
                //        {    
                //            m_activeIdx = iter.key();
                //            m_activeText = iter.value();
                //            m_unfold = false;
                //        
                //            if(m_variable)
                //                *m_variable = m_activeIdx;
                //        }
                //    }

                //    i++;
                //}
            }
        }



        return m_mouseClick;
    }

    void onMouseRelease()
    {
        m_mouseClick = false;
    }

    int activeIdx() const
    {
        return m_activeIdx;
    }

    void setActiveIdx(int active)
    {
        //if(m_items.size() == 0)
        //    return;

        //m_activeIdx = active;

        //QMap<int, QString>::iterator iter = m_items.find(active);
        //if(iter == m_items.end())
        //{
        //    m_activeIdx = 0;
        //    m_activeText = m_items[m_activeIdx];
        //}
        //else
        //{
        //    m_activeIdx = iter.key();
        //    m_activeText = iter.value();
        //}

        //if(m_variable)
        //    *m_variable = m_activeIdx;
    }

    void setColor(const vec4 &color)
    {
        m_color = color;
    }

    //void ComboBox::setVariable(int *variable)
    //{
    //    m_variable = variable;
    //}

    void setPosition(uint x, uint y)
    {
	    m_posX = x;
	    m_posY = y;
    }

    void addItem(int id, string name)
    {
        //m_items.insert(id, name);
        //m_activeIdx = id;
        //m_activeText = name;
    }

    vec2 position()
    {
        return vec2((float)m_posX, (float)m_posY);
    }

    vec2 dimensions()
    {
        return vec2(float(m_width), (float)m_height);
    }

private:  
    uint m_width;
    uint m_height;
    uint m_posX;
    uint m_posY;

    string m_text;
    vec4 m_color;

    bool m_unfold;
    bool m_mouseClick;
    int *m_variable;   
    int m_activeIdx;
    string m_activeText;
    int rowHeight;
    int m_hooverId;

    //QMap<int, QString> m_items;
};

template <class T> class Slider : public GUIElement
{
public:
    Slider(uint px, uint py, uint w, uint h, string text)
    : m_width(w), 
        m_height(h),
        m_posX(px),
        m_posY(py),
        m_mouseClick(false),
        m_text(text),
        m_minValue(static_cast<T>(0.0f)),
        m_maxValue(static_cast<T>(1.0f)),
        m_value(static_cast<T>(0.0f)),
        m_valueRange(static_cast<T>(0.0f)),
        m_variable(NULL),
        m_color(1.0f, 1.0f, 1.0f, 1.0f)
    {
	    m_range = (float)m_width;
	    m_percent = 1.0f;

	    m_valueRange = m_maxValue - m_minValue;
    }

    ~Slider() {};

    void render()
    {
        glEnable2D();

        float x1 = float(m_posX);
        float x2 = float(m_posX + m_width);
        float y1 = float(m_posY);
        float y2 = float(m_posY + m_height);
    
        glColor4f(m_color.x, m_color.y, m_color.z,  m_color.w);
    
        glBegin(GL_LINES);
            glVertex2f(x1, y1);
            glVertex2f(x2, y1);

            glVertex2f(x1-1, y2);
            glVertex2f(x2, y2);
        
            glVertex2f(x1, y1);
            glVertex2f(x1, y2);

            glVertex2f(x2, y1);
            glVertex2f(x2, y2);
        glEnd();

        float qx1 = float(m_posX + 1);
        float qx2 = float(m_posX + 1 + (m_width-3) * m_percent);
        float qy1 = float(m_posY + 2);
        float qy2 = float(m_posY + m_height - 1);    
  
        glBegin(GL_QUADS);
            glVertex2f(qx1, qy1);
            glVertex2f(qx1, qy2);
            glVertex2f(qx2, qy2);
            glVertex2f(qx2, qy1);
        glEnd();

        glDisable2D();

        string sliderText = m_text;

        std::ostringstream stream;
        stream << m_value;
        std::string str = sliderText + stream.str();
        renderString(str.c_str(), m_posX-2, m_posY-5, m_color.x, m_color.y, m_color.z, m_color.w, GLUT_BITMAP_HELVETICA_12);
    }

    bool onMouseClick(uint mx, uint my)
    {
        float x1 = float(m_posX);
        float x2 = float(m_posX + m_width);
        float y1 = float(m_posY);
        float y2 = float(m_posY + m_height);

        m_mouseClick = false;

        int curPosX = 0;

        if(mx > x1 && mx < x2)
        {
            if(my > y1 && my < y2)
            {    
                m_mouseClick = true;

                curPosX = mx - m_posX;
                m_percent = (float)curPosX / m_range;

                m_value = static_cast<T>( m_minValue + m_valueRange * m_percent );
            
                if(m_variable)
                    *m_variable = m_value;
            }
        }

        return m_mouseClick;
    }

    void onMouseMove(uint mx, uint my)
    {
        float x1 = float(m_posX);
        float x2 = float(m_posX + m_width);

        if(m_mouseClick)
        {
            int curPosX = 0;    

            if(mx > x1 && mx < x2)
            {
                curPosX = mx - m_posX;
                m_percent = (float)curPosX / m_range;
            }

            if(mx < x1)
            {
                m_percent = 0.0f;
            }
            if(mx > x2)
            {
                m_percent = 1.0f;
            }

            m_value = static_cast<T>( m_minValue + m_valueRange * m_percent );
		    //m_value = static_cast<T>( 1024* int(100 * m_percent) );
		    //if(m_value < m_minValue) m_value = m_minValue;
        
            if(m_variable)
                *m_variable = m_value;
        }
    }

    void onMouseRelease()
    {
        m_mouseClick = false;
    }

    void setPosition(uint x, uint y)
    {
	    m_posX = x;
	    m_posY = y;
    }

    void setValue(T value)
    {
        m_value = value;

        if(m_value < m_minValue)
            m_value = m_minValue;

        if(m_value > m_maxValue)
            m_value = m_maxValue;

        if(m_variable)
            *m_variable = value;

        if(value < static_cast<T>(0.0f))
	    {
            m_percent = (abs(static_cast<float>(m_minValue)) - fabs(static_cast<float>(m_value))) / static_cast<float>(m_valueRange);
	    } else {
            m_percent = fabs(static_cast<float>(m_value) - static_cast<float>(m_minValue)) / static_cast<float>(m_valueRange);
	    }
    }
   
    void setRange(T min, T max)
    {
        if(min < max && max > static_cast<T>(0.0f))
        {
            m_minValue = min;
            m_maxValue = max;   

            m_valueRange = max - min;
            m_value = (T)(min + m_valueRange * m_percent);

            if(m_variable)
                *m_variable = m_value;
        }
        else
        {
            float tmpMin = static_cast<float>(min);
            float tmpMax = static_cast<float>(max);

            if(tmpMin < tmpMax)
            {
                m_minValue = min;
                m_maxValue = max;        
            
                m_valueRange = static_cast<T>( tmpMax - tmpMin );
                m_value      = static_cast<T>( m_minValue + m_valueRange * m_percent );

                if(m_variable)
                    *m_variable = m_value;
            }
            else
            {
                m_minValue = max;
                m_maxValue = min;

                m_valueRange = static_cast<T>( tmpMin - tmpMax );
                m_value      = static_cast<T>( m_minValue + m_valueRange * m_percent );
            
                if(m_variable)
                    *m_variable = m_value;
            }
        }
    }

    void setColor(const vec4 &color)
    {
        m_color = color;
    }

    void setVariable(T *variable)
    {
        m_variable = variable;
    }

    T value() const
    {
        return m_value;
    }

    vec2 position()
    {
	    return vec2(m_posX, m_posY);
    }

    vec2 dimensions()
    {
	    return vec2(m_width, m_height);
    }

private:
    uint m_width;
    uint m_height;
    uint m_posX;
    uint m_posY;

    string m_text;

    float m_percent;
    float m_range;
    T m_valueRange;
    T m_value;
    T m_minValue;
    T m_maxValue;
	vec4 m_color;

    bool m_mouseClick;

    T *m_variable;   
};

class CheckBox : public GUIElement
{
public:
    CheckBox(uint px, uint py, uint w, uint h, string text)
    : m_width(w), 
      m_height(h),
      m_posX(px),
      m_posY(py),
      m_mouseClick(false),
      m_text(text),
      m_variable(NULL),
      m_state(false),
      m_color(1.0f, 1.0f, 1.0f, 1.0f)
    {    
    }

    ~CheckBox()
    {
    }

    void render()
    {
        glEnable2D();

        float x1 = float(m_posX);
        float x2 = float(m_posX + m_width);
        float y1 = float(m_posY);
        float y2 = float(m_posY + m_height);
    
        glColor4f(m_color.x, m_color.y, m_color.z,  m_color.w);
    
        glBegin(GL_LINES);
            glVertex2f(x1, y1);
            glVertex2f(x2, y1);

            glVertex2f(x1-1, y2);
            glVertex2f(x2, y2);
        
            glVertex2f(x1, y1);
            glVertex2f(x1, y2);

            glVertex2f(x2, y1);
            glVertex2f(x2, y2);
        glEnd(); 

        if(m_state)
        {
            glBegin(GL_QUADS);
                glVertex2f(x1+1, y1+2);
                glVertex2f(x1+1, y2-1);
                glVertex2f(x2-2, y2-1);            
                glVertex2f(x2-2, y1+2);
            glEnd();
        }

        glDisable2D();

        string sliderText = m_text;
        renderString(sliderText.c_str(), m_posX + m_width + 3, m_posY + 12, m_color.x, m_color.y, m_color.z, m_color.w, GLUT_BITMAP_HELVETICA_12);
    }

    void onMouseMove(uint mx, uint my)
    {
    }

    bool onMouseClick(uint mx, uint my)
    {
        float x1 = float(m_posX);
        float x2 = float(m_posX + m_width);
        float y1 = float(m_posY);
        float y2 = float(m_posY + m_height);

        m_mouseClick = false;

        if(mx > x1 && mx < x2)
        {
            if(my > y1 && my < y2)
            {   
                m_state = !m_state;
            
                if(m_variable)
                    (*m_variable) = m_state;

                m_mouseClick = true;
            }
        }

        return m_mouseClick;
    }

    void onMouseRelease()
    {
        m_mouseClick = false;
    }

    bool state() const
    {
        return m_state;
    }

    void setState(bool value)
    {
        m_state = value;
    
        if(m_variable)
            *m_variable = value;
    }

    void setColor(const vec4 &color)
    {
        m_color = color;
    }

    void setVariable(bool *variable)
    {
        m_variable = variable;
    }

    void setPosition(uint x, uint y)
    {
	    m_posX = x;
	    m_posY = y;
    }

    vec2 position()
    {
        return vec2((float)m_posX, (float)m_posY);
    }

    vec2 dimensions()
    {
        return vec2((float)m_width, (float)m_height);
    }

private:  
    uint m_width;
    uint m_height;
    uint m_posX;
    uint m_posY;

    string m_text;

    vec4 m_color;

    bool m_mouseClick;

    bool m_state;
    bool *m_variable;
};

class Texture;

class Button : public GUIElement
{
public:
    Button(int posX, int posY, int width, int height, string texPath, string text)
    : m_posX(posX),
      m_posY(posY),
      m_width(width),
      m_height(height),
      m_text(text),
      m_mouseClick(false),
      m_color(0, 0, 0, 1),
      m_texture(nullptr),
      m_variable(nullptr)
    {
        //QImage imgTexNormal(texPath);
        //m_texture = new Texture(imgTexNormal);
    }

    ~Button()
    {
        delete m_texture;
    }

    void render()
    {   
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);
   
        glEnable2D();

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, m_texture->id());

        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 

        glColor4f(m_color.x, m_color.y, m_color.z, m_color.w);
        glPushMatrix();
 
        glTranslatef((float)m_posX, (float)m_posY, 0.0f);
        glScalef((float)m_width, (float)m_height, 0.0f);

        glBegin(GL_QUADS);   
            glTexCoord2f(0.0f, 0.0f);
            glVertex3f(0.0f, 1.0f, 0.0f);

            glTexCoord2f(1.0f, 0.0f);
            glVertex3f(1.0f, 1.0f, 0.0f);                        

            glTexCoord2f(1.0f, 1.0f);
            glVertex3f(1.0f, 0.0, 0.0f);

            glTexCoord2f(0.0f, 1.0f);
            glVertex3f(0.0f, 0.0f, 0.0f);
        glEnd();
        glPopMatrix();      

        glDisable(GL_TEXTURE_2D);

        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE0);

        glDisable2D();

        //renderString(m_text.c_str(), m_posX, m_posY + 50, Vector4(1, 1, 1, 1), GLUT_BITMAP_HELVETICA_10);

        glPopClientAttrib();
        glPopAttrib();
    }

    void onMouseMove(uint mx, uint my)
    {
    }

    bool onMouseClick(uint mx, uint my)
    {
        float x1 = float(m_posX);
        float x2 = float(m_posX + m_width);
        float y1 = float(m_posY);
        float y2 = float(m_posY + m_height);

        m_mouseClick = false;

        if(mx > x1 && mx < x2)
        {
            if(my > y1 && my < y2)
            {             
                m_mouseClick = true;
                if(m_variable)
                    (*m_variable) = true;
            }
        }

        return m_mouseClick;
    }

    void onMouseRelease()
    {
        m_mouseClick = false;
    }

    void setPosition(uint x, uint y)
    {
	    m_posX = x;
	    m_posY = y;
    }

    vec2 position()
    {
        return vec2((float)m_posX, (float)m_posY);
    }

    vec2 dimensions()
    {
        return vec2((float)m_width, (float)m_height);
    }

    void setVariable(bool *variable)
    {
        m_variable = variable;
    }

    void setColor(vec4 color)
    {
        m_color = color;
    }

private:
    int m_width;
    int m_height;
    int m_posX;
    int m_posY;

    Texture *m_texture;
   
    string m_text;

    bool m_mouseClick;

    bool *m_variable;
    vec4 m_color;
};


class GUI
{

private:
    enum Slots
    {
        TAB1 = 0,
        TAB2,
        TAB3,
        TAB4,
        TAB5
    };

public:
   GUI(CameraManager *cameraManager, Scene *m_scene);
   ~GUI();

   void render();
   void setFontColor(const vec4 &color);
   void resize(int width, int height);

   bool onMouseClick();
   void onMouseMove();
   void onMouseRelease(); 

private:
	void initUIElements();    
    void renderTabs();
    void renderBars();

    void initTab1(Slots slot);
    void initTab2(Slots slot);
    void initTab3(Slots slot);
    void initTab4(Slots slot);
    void initTab5(Slots slot);

    void setupCheckBox(int slot, int x, int y, string &name, bool *var);
    void setupFloatSlider(int slot, int x, int y, float rangeMin, float rangeMax, const string &name, float *var);
    void setupIntSlider(int slot, int x, int y, int rangeMin, int rangeMax, const string &name, int *var);

private:
	uint m_width;
	uint m_height;

	uint m_fpsCount;
	uint m_fps;

    int m_curTab;

    HPTimer m_timer;
    double m_oldTime;

    CameraManager *m_cameraManager;  
	Scene *m_scene;

	vec4 m_fontColor;

	VertexBufferObject *m_vboQuad;

    vector<vector<GUIElement*>> m_guiAll;
    vector<string> m_tabNames;

    HPTimer m_fpsTimer;
};

#endif

